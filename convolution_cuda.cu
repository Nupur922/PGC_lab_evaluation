#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// ------------------------------------------------------------
// Problem configuration
// ------------------------------------------------------------

const int N = 4096;
const int K = 3;

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t error = call;                                \
    if (error != cudaSuccess)                                \
    {                                                        \
        cerr << "CUDA Error: "                              \
             << cudaGetErrorString(error)                   \
             << " at line " << __LINE__ << endl;            \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

// ------------------------------------------------------------
// Generate the SAME deterministic input as CPU
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// SAME 3x3 kernel as CPU
// ------------------------------------------------------------

void initializeKernel(vector<float>& kernel)
{
    kernel =
    {
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f
    };
}

// ------------------------------------------------------------
// CUDA convolution kernel
// ------------------------------------------------------------

__global__ void convolutionCUDA(
    const float* input,
    const float* kernel,
    float* output,
    int inputSize,
    int kernelSize)
{
    // Global output coordinates
    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int outputSize =
        inputSize - kernelSize + 1;

    // Make sure thread corresponds to valid output element
    if (row < outputSize && col < outputSize)
    {
        float sum = 0.0f;

        // Apply convolution kernel
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

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    const int outputSize = N - K + 1;

    size_t inputBytes =
        static_cast<size_t>(N) * N * sizeof(float);

    size_t kernelBytes =
        static_cast<size_t>(K) * K * sizeof(float);

    size_t outputBytes =
        static_cast<size_t>(outputSize) *
        outputSize *
        sizeof(float);

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    vector<float> input(N * N);
    vector<float> kernel(K * K);
    vector<float> output(outputSize * outputSize);

    initializeInput(input);
    initializeKernel(kernel);

    // --------------------------------------------------------
    // Display GPU information
    // --------------------------------------------------------

    cudaDeviceProp deviceProperties;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &deviceProperties,
            0
        )
    );

    cout << "====================================================\n";
    cout << "             CUDA 2D CONVOLUTION - GPU\n";
    cout << "====================================================\n";

    cout << "GPU              : "
         << deviceProperties.name << "\n";

    cout << "Compute capability: "
         << deviceProperties.major
         << "."
         << deviceProperties.minor
         << "\n";

    cout << "GPU memory       : "
         << fixed << setprecision(2)
         << deviceProperties.totalGlobalMem /
                (1024.0 * 1024.0 * 1024.0)
         << " GB\n";

    cout << "Input size       : "
         << N << " x " << N << "\n";

    cout << "Kernel size      : "
         << K << " x " << K << "\n";

    cout << "Output size      : "
         << outputSize << " x "
         << outputSize << "\n";

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // CPU -> GPU transfers
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // CUDA launch configuration
    // --------------------------------------------------------

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (outputSize + blockSize.x - 1)
            / blockSize.x,

        (outputSize + blockSize.y - 1)
            / blockSize.y
    );

    cout << "Block size       : "
         << blockSize.x << " x "
         << blockSize.y << "\n";

    cout << "Grid size        : "
         << gridSize.x << " x "
         << gridSize.y << "\n";

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    convolutionCUDA<<<gridSize, blockSize>>>(
        d_input,
        d_kernel,
        d_output,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // CUDA event timing
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    CUDA_CHECK(
        cudaEventRecord(start)
    );

    convolutionCUDA<<<gridSize, blockSize>>>(
        d_input,
        d_kernel,
        d_output,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    float gpuTime = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuTime,
            start,
            stop
        )
    );

    // --------------------------------------------------------
    // GPU -> CPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            output.data(),
            d_output,
            outputBytes,
            cudaMemcpyDeviceToHost
        )
    );

    // --------------------------------------------------------
    // Display results
    // --------------------------------------------------------

    cout << fixed << setprecision(3);

    cout << "\nGPU kernel execution time : "
         << gpuTime
         << " ms\n";

    cout << "\nFirst 10 output values:\n";

    for (int i = 0; i < 10; i++)
    {
        cout << output[i] << " ";
    }

    cout << "\n";

    cout << "\n====================================================\n";

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}