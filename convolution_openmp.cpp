#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <omp.h>

using namespace std;

// ------------------------------------------------------------
// Problem configuration
// ------------------------------------------------------------

const int N = 4096;
const int K = 3;

// ------------------------------------------------------------
// Generate deterministic input
// ------------------------------------------------------------

void initializeInput(vector<float>& input)
{
    mt19937 generator(42);
    uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (float& value : input)
    {
        value = distribution(generator);
    }
}

// ------------------------------------------------------------
// 3x3 convolution kernel
// ------------------------------------------------------------

void initializeKernel(vector<float>& kernel)
{
    kernel = {
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f,
         1.0f,  0.0f, -1.0f
    };
}

// ------------------------------------------------------------
// OpenMP 2D convolution
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    const int outputSize = N - K + 1;

    vector<float> input(N * N);
    vector<float> kernel(K * K);
    vector<float> output(outputSize * outputSize);

    initializeInput(input);
    initializeKernel(kernel);

    cout << "====================================================\n";
    cout << "          OpenMP 2D CONVOLUTION - CPU\n";
    cout << "====================================================\n";

    cout << "Input size       : "
         << N << " x " << N << "\n";

    cout << "Kernel size      : "
         << K << " x " << K << "\n";

    cout << "Output size      : "
         << outputSize << " x " << outputSize << "\n";

    cout << "Maximum threads  : "
         << omp_get_max_threads() << "\n";

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    convolutionOpenMP(input, kernel, output);

    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    double start = omp_get_wtime();

    convolutionOpenMP(input, kernel, output);

    double end = omp_get_wtime();

    double executionTime = (end - start) * 1000.0;

    cout << fixed << setprecision(3);

    cout << "\nCPU execution time : "
         << executionTime << " ms\n";

    // --------------------------------------------------------
    // Display sample results
    // --------------------------------------------------------

    cout << "\nFirst 10 output values:\n";

    for (int i = 0; i < 10; i++)
    {
        cout << output[i] << " ";
    }

    cout << "\n";

    cout << "\n====================================================\n";

    return 0;
}