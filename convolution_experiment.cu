#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t error = call;                                \
    if (error != cudaSuccess)                                \
    {                                                        \
        cerr << "CUDA ERROR: "                              \
             << cudaGetErrorString(error)                   \
             << " at line " << __LINE__ << endl;            \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

// ============================================================
// CONFIGURATION
// ============================================================

const int K = 3;
const int ITERATIONS = 100;

// ============================================================
// INPUT GENERATION
// ============================================================

void initializeInput(vector<float>& input)
{
    mt19937 generator(42);

    uniform_real_distribution<float> distribution(
        0.0f,
        1.0f
    );

    for (float& value : input)
    {
        value = distribution(generator);
    }
}

// ============================================================
// 3x3 KERNEL
// ============================================================

void initializeKernel(vector<float>& kernel)
{
    kernel =
    {
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f
    };
}

// ============================================================
// OPENMP CPU CONVOLUTION
// ============================================================

void convolutionOpenMP(
    const vector<float>& input,
    const vector<float>& kernel,
    vector<float>& output,
    int N)
{
    const int outputSize = N - K + 1;

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < outputSize; row++)
    {
        for (int col = 0; col < outputSize; col++)
        {
            float sum = 0.0f;

            for (int kr = 0; kr < K; kr++)
            {
                for (int kc = 0; kc < K; kc++)
                {
                    int inputRow = row + kr;
                    int inputCol = col + kc;

                    sum +=
                        input[inputRow * N + inputCol] *
                        kernel[kr * K + kc];
                }
            }

            output[row * outputSize + col] = sum;
        }
    }
}

// ============================================================
// CUDA GPU CONVOLUTION
// ============================================================

__global__ void convolutionCUDA(
    const float* input,
    const float* kernel,
    float* output,
    int inputSize,
    int kernelSize)
{
    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int outputSize =
        inputSize - kernelSize + 1;

    if (row < outputSize && col < outputSize)
    {
        float sum = 0.0f;

        for (int kr = 0; kr < kernelSize; kr++)
        {
            for (int kc = 0; kc < kernelSize; kc++)
            {
                int inputRow = row + kr;
                int inputCol = col + kc;

                sum +=
                    input[inputRow * inputSize + inputCol] *
                    kernel[kr * kernelSize + kc];
            }
        }

        output[row * outputSize + col] = sum;
    }
}

// ============================================================
// RESULT COMPARISON
// ============================================================

void compareResults(
    const vector<float>& cpu,
    const vector<float>& gpu,
    double& maxError,
    double& meanError,
    long long& mismatches)
{
    double totalError = 0.0;
    maxError = 0.0;
    mismatches = 0;

    const double tolerance = 1e-5;

    for (size_t i = 0; i < cpu.size(); i++)
    {
        double error =
            fabs(
                static_cast<double>(cpu[i]) -
                static_cast<double>(gpu[i])
            );

        totalError += error;

        maxError =
            max(maxError, error);

        if (error > tolerance)
        {
            mismatches++;
        }
    }

    meanError =
        totalError / cpu.size();
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    // --------------------------------------------------------
    // Command-line arguments
    // --------------------------------------------------------

    // Display the formatted report by default. Use --quiet to suppress it
    // when running large automated loops; CSV logging remains enabled.
    const bool quiet = (argc == 4 && string(argv[3]) == "--quiet");

    if (argc != 3 && !quiet)
    {
        cerr << "Usage: convolution_experiment.exe <matrix_size> <threads> [--quiet]\n";
        return 1;
    }

    int N = stoi(argv[1]);
    int requestedThreads = stoi(argv[2]);

    if (requestedThreads < 1)
    {
        cerr << "Thread count must be at least 1.\n";
        return 1;
    }

    if (N < K)
    {
        cerr << "Matrix size must be at least 3.\n";
        return 1;
    }

    // --------------------------------------------------------
    // OpenMP configuration
    // --------------------------------------------------------

    omp_set_dynamic(0);
    omp_set_num_threads(requestedThreads);

    int actualThreads =
        omp_get_max_threads();

    // --------------------------------------------------------
    // Matrix sizes
    // --------------------------------------------------------

    const int outputSize =
        N - K + 1;

    const size_t inputElements =
        static_cast<size_t>(N) * N;

    const size_t outputElements =
        static_cast<size_t>(outputSize) *
        outputSize;

    const size_t inputBytes =
        inputElements * sizeof(float);

    const size_t kernelBytes =
        K * K * sizeof(float);

    const size_t outputBytes =
        outputElements * sizeof(float);

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    vector<float> input(inputElements);
    vector<float> kernel(K * K);

    vector<float> cpuOutput(outputElements);
    vector<float> gpuOutput(outputElements);

    initializeInput(input);
    initializeKernel(kernel);

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    // ========================================================
    // CPU WARM-UP
    // ========================================================

    convolutionOpenMP(
        input,
        kernel,
        cpuOutput,
        N
    );

    // ========================================================
    // CPU BENCHMARK
    // ========================================================

    double cpuStart =
        omp_get_wtime();

    for (int i = 0; i < ITERATIONS; i++)
    {
        convolutionOpenMP(
            input,
            kernel,
            cpuOutput,
            N
        );
    }

    double cpuEnd =
        omp_get_wtime();

    double cpuAverageTime =
        ((cpuEnd - cpuStart) * 1000.0)
        / ITERATIONS;

    // ========================================================
    // GPU MEMORY
    // ========================================================

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            inputBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_kernel,
            kernelBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            outputBytes
        )
    );

    // ========================================================
    // COPY INPUT TO GPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            input.data(),
            inputBytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_kernel,
            kernel.data(),
            kernelBytes,
            cudaMemcpyHostToDevice
        )
    );

    // ========================================================
    // CUDA CONFIGURATION
    // ========================================================

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (outputSize + blockSize.x - 1)
            / blockSize.x,

        (outputSize + blockSize.y - 1)
            / blockSize.y
    );

    // ========================================================
    // GPU WARM-UP
    // ========================================================

    convolutionCUDA<<<
        gridSize,
        blockSize
    >>>(
        d_input,
        d_kernel,
        d_output,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================
    // GPU KERNEL BENCHMARK
    // ========================================================

    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    CUDA_CHECK(
        cudaEventRecord(kernelStart)
    );

    for (int i = 0; i < ITERATIONS; i++)
    {
        convolutionCUDA<<<
            gridSize,
            blockSize
        >>>(
            d_input,
            d_kernel,
            d_output,
            N,
            K
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaEventRecord(kernelStop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(kernelStop)
    );

    float gpuTotalKernelTime = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuTotalKernelTime,
            kernelStart,
            kernelStop
        )
    );

    double gpuAverageKernelTime =
        gpuTotalKernelTime / ITERATIONS;

    // ========================================================
    // CORRECT END-TO-END GPU TIMING
    //
    // ONE complete operation:
    // Host -> Device
    // Device computation
    // Device -> Host
    // ========================================================

    cudaEvent_t e2eStart;
    cudaEvent_t e2eStop;

    CUDA_CHECK(cudaEventCreate(&e2eStart));
    CUDA_CHECK(cudaEventCreate(&e2eStop));

    CUDA_CHECK(
        cudaEventRecord(e2eStart)
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            input.data(),
            inputBytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_kernel,
            kernel.data(),
            kernelBytes,
            cudaMemcpyHostToDevice
        )
    );

    convolutionCUDA<<<
        gridSize,
        blockSize
    >>>(
        d_input,
        d_kernel,
        d_output,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpy(
            gpuOutput.data(),
            d_output,
            outputBytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaEventRecord(e2eStop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(e2eStop)
    );

    float gpuEndToEndTime = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuEndToEndTime,
            e2eStart,
            e2eStop
        )
    );

    // ========================================================
    // NUMERICAL VERIFICATION
    // ========================================================

    double maxError;
    double meanError;
    long long mismatches;

    compareResults(
        cpuOutput,
        gpuOutput,
        maxError,
        meanError,
        mismatches
    );

    // ========================================================
    // SPEEDUP
    // ========================================================

    double kernelSpeedup =
        cpuAverageTime /
        gpuAverageKernelTime;

    double endToEndSpeedup =
        cpuAverageTime /
        gpuEndToEndTime;

    // ========================================================
    // FORMATTED TERMINAL REPORT (shown by default)
    // ========================================================

    if (!quiet)
    {
        cout << fixed << setprecision(4);
        cout << "\n============================================================\n";
        cout << "               CONVOLUTION EXPERIMENT\n";
        cout << "============================================================\n";
        cout << "Matrix size          : " << N << " x " << N << "\n";
        cout << "Kernel               : " << K << " x " << K << "\n";
        cout << "Output size          : " << outputSize << " x " << outputSize << "\n";
        cout << "OpenMP threads       : " << actualThreads << "\n";
        cout << "CUDA block           : " << blockSize.x << " x " << blockSize.y << "\n";
        cout << "CUDA grid            : " << gridSize.x << " x " << gridSize.y << "\n\n";

        cout << "PERFORMANCE\n";
        cout << "------------------------------------------------------------\n";
        cout << "OpenMP CPU           : " << cpuAverageTime << " ms\n";
        cout << "CUDA kernel          : " << gpuAverageKernelTime << " ms\n";
        cout << "CUDA end-to-end      : " << gpuEndToEndTime << " ms\n";
        cout << "Kernel speedup       : " << kernelSpeedup << " x\n";
        cout << "End-to-end speedup   : " << endToEndSpeedup << " x\n\n";

        cout << "NUMERICAL VERIFICATION\n";
        cout << "------------------------------------------------------------\n";
        cout << scientific << setprecision(6);
        cout << "Maximum error        : " << maxError << "\n";
        cout << "Mean error           : " << meanError << "\n";
        cout << fixed << setprecision(0);
        cout << "Mismatches           : " << mismatches << "\n";
        cout << "Status               : " << (mismatches == 0 ? "PASS" : "FAIL") << "\n";
        cout << "============================================================\n";
    }

    // ========================================================
    // APPEND RESULTS TO CSV
    // Saved in the program's current working directory.
    // ========================================================

    const string csvFile = "convolution_results.csv";
    bool writeHeader = false;

    {
        ifstream checkFile(csvFile, ios::binary);
        writeHeader = !checkFile.good() || checkFile.peek() == ifstream::traits_type::eof();
    }

    ofstream csv(csvFile, ios::out | ios::app);
    if (!csv.is_open())
    {
        cerr << "ERROR: Could not open '" << csvFile
             << "' for writing. Close it in Excel if it is open, then retry.\n";
        return 1;
    }

    csv << fixed << setprecision(9);

    if (writeHeader)
    {
        csv << "MatrixSize,Threads,CPU_ms,GPU_Kernel_ms,GPU_EndToEnd_ms,"
               "Kernel_Speedup,EndToEnd_Speedup,Max_Error,Mean_Error,Mismatches\n";
    }

    csv << N << ','
        << actualThreads << ','
        << cpuAverageTime << ','
        << gpuAverageKernelTime << ','
        << gpuEndToEndTime << ','
        << kernelSpeedup << ','
        << endToEndSpeedup << ','
        << maxError << ','
        << meanError << ','
        << mismatches << '\n';

    csv.flush();
    if (!csv.good())
    {
        cerr << "ERROR: Failed while writing to '" << csvFile << "'.\n";
        return 1;
    }
    csv.close();

    // ========================================================
    // CLEANUP
    // ========================================================

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    CUDA_CHECK(cudaEventDestroy(e2eStart));
    CUDA_CHECK(cudaEventDestroy(e2eStop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}