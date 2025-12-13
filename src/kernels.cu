
namespace
{
    __global__ void saxpy_kernel(int n, float a, float* x, float* y)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a*x[i] + y[i];
    }
}

void saxpy(int n, float a, float* x, float* y)
{
    float* d_x{};
    float* d_y{};
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}


