#include <iostream>
#include "kernels.cuh"

int main()
{
    int N = 1<<20;
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    saxpy(N, 2.0f, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = std::max(maxError, abs(y[i]-4.0f));
    }
    printf("Max error: %f\n", maxError);

    free(x);
    free(y);
}
