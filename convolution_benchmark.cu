#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

// ============================================================
// CONFIGURATION
// ============================================================

const int N = 4096;
const int K = 3;

const int ITERATIONS = 100;

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t error = call;                                \
    if (error != cudaSuccess)                                \
    {                                                        \
        cerr << "\nCUDA ERROR: "                             \
             << cudaGetErrorString(error)                   \
             << "\nLine: " << __LINE__ << endl;             \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

// ============================================================
// INITIALIZE INPUT
// Same input for CPU and GPU
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
// INITIALIZE 3x3 KERNEL
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
    vector<float>& output)
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
// NUMERICAL COMPARISON
// ============================================================

void compareResults(
    const vector<float>& cpu,
    const vector<float>& gpu)
{
    double totalError = 0.0;
    double maxError = 0.0;

    size_t mismatches = 0;

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

    double meanError =
        totalError / cpu.size();

    cout << "\n";
    cout << "NUMERICAL VERIFICATION\n";
    cout << "------------------------------------------------------------\n";

    cout << scientific << setprecision(6);

    cout << "Maximum absolute error : "
         << maxError << "\n";

    cout << "Mean absolute error    : "
         << meanError << "\n";

    cout << fixed << setprecision(0);

    cout << "Mismatches             : "
         << mismatches << "\n";

    if (mismatches == 0)
    {
        cout << "Verification status    : PASS\n";
    }
    else
    {
        cout << "Verification status    : FAIL\n";
    }
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    const int outputSize = N - K + 1;

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
    // HOST MEMORY
    // --------------------------------------------------------

    vector<float> input(inputElements);
    vector<float> kernel(K * K);

    vector<float> cpuOutput(outputElements);
    vector<float> gpuOutput(outputElements);

    initializeInput(input);
    initializeKernel(kernel);

    // --------------------------------------------------------
    // GPU INFORMATION
    // --------------------------------------------------------

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    // --------------------------------------------------------
    // HEADER
    // --------------------------------------------------------

    cout << "\n";
    cout << "============================================================\n";
    cout << "             2D CONVOLUTION CPU vs GPU\n";
    cout << "============================================================\n";

    cout << "\nHARDWARE\n";
    cout << "------------------------------------------------------------\n";

    cout << "CPU              : Intel Core i5-13450HX\n";

    cout << "OpenMP threads   : "
         << omp_get_max_threads()
         << "\n";

    cout << "GPU              : "
         << prop.name
         << "\n";

    cout << "Compute capability: "
         << prop.major
         << "."
         << prop.minor
         << "\n";

    cout << "GPU memory       : "
         << fixed << setprecision(2)
         << prop.totalGlobalMem /
                (1024.0 * 1024.0 * 1024.0)
         << " GB\n";

    cout << "\nPROBLEM\n";
    cout << "------------------------------------------------------------\n";

    cout << "Input            : "
         << N << " x " << N
         << "\n";

    cout << "Kernel           : "
         << K << " x " << K
         << "\n";

    cout << "Output           : "
         << outputSize << " x "
         << outputSize
         << "\n";

    cout << "Data type        : float\n";

    cout << "Iterations       : "
         << ITERATIONS
         << "\n";

    // ========================================================
    // CPU WARM-UP
    // ========================================================

    convolutionOpenMP(
        input,
        kernel,
        cpuOutput
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
            cpuOutput
        );
    }

    double cpuEnd =
        omp_get_wtime();

    double cpuTotalTime =
        (cpuEnd - cpuStart) * 1000.0;

    double cpuAverageTime =
        cpuTotalTime / ITERATIONS;

    // ========================================================
    // GPU MEMORY ALLOCATION
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
    // GPU END-TO-END TRANSFER + COMPUTATION TIMING
    // ========================================================

    cudaEvent_t e2eStart;
    cudaEvent_t e2eStop;

    CUDA_CHECK(cudaEventCreate(&e2eStart));
    CUDA_CHECK(cudaEventCreate(&e2eStop));

    CUDA_CHECK(
        cudaEventRecord(e2eStart)
    );

    // Host -> Device
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
    // CUDA LAUNCH CONFIGURATION
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

    // --------------------------------------------------------
    // GPU -> HOST
    // --------------------------------------------------------

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
    // PERFORMANCE RESULTS
    // ========================================================

    cout << "\nPERFORMANCE\n";
    cout << "------------------------------------------------------------\n";

    cout << fixed << setprecision(4);

    cout << "OpenMP CPU average       : "
         << cpuAverageTime
         << " ms\n";

    cout << "CUDA kernel average     : "
         << gpuAverageKernelTime
         << " ms\n";

    cout << "CUDA end-to-end time    : "
         << gpuEndToEndTime
         << " ms\n";

    double kernelSpeedup =
        cpuAverageTime /
        gpuAverageKernelTime;

    double endToEndSpeedup =
        cpuAverageTime /
        gpuEndToEndTime;

    cout << "\n";

    cout << "Kernel speedup           : "
         << kernelSpeedup
         << " x\n";

    cout << "End-to-end speedup      : "
         << endToEndSpeedup
         << " x\n";

    cout << "\n";

    cout << "CUDA block size         : "
         << blockSize.x
         << " x "
         << blockSize.y
         << "\n";

    cout << "CUDA grid size          : "
         << gridSize.x
         << " x "
         << gridSize.y
         << "\n";

    // ========================================================
    // NUMERICAL VERIFICATION
    // ========================================================

    compareResults(
        cpuOutput,
        gpuOutput
    );

    // ========================================================
    // SAMPLE OUTPUT
    // ========================================================

    cout << "\nSAMPLE OUTPUT\n";
    cout << "------------------------------------------------------------\n";

    cout << fixed << setprecision(6);

    cout << "CPU: ";

    for (int i = 0; i < 10; i++)
    {
        cout << cpuOutput[i] << " ";
    }

    cout << "\nGPU: ";

    for (int i = 0; i < 10; i++)
    {
        cout << gpuOutput[i] << " ";
    }

    cout << "\n";

    cout << "\n============================================================\n";

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