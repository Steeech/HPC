#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <cmath>
#define BLOCK_SIZE  16          
#define N           1024        

__global__ void matMult(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x;     
    int by = blockIdx.y;
    int tx = threadIdx.x;      
    int ty = threadIdx.y;
    float sum = 0.0f;           
    int ia = n * BLOCK_SIZE * by + n * ty;  
    int ib = BLOCK_SIZE * bx + tx;
    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];
    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic + n * ty + tx] = sum;
}

int main(int argc, char* argv[])
{
    int numBytes = N * N * sizeof(float);
    float* a = new float[N * N];
    float* b = new float[N * N];
    float* c = new float[N * N];
    srand(time(0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int	k = N * i + j;

            a[k] = rand() % 10;
            b[k] = rand() % 10;
        }

    float* adev = NULL;
    float* bdev = NULL;
    float* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMult << <blocks, threads >> > (adev, bdev, N, cdev);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);

    float* c_cpu = new float[N * N];
    unsigned int cpu_start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
        {
            float summ = 0;
            for (int k = 0; k < N; k++) {
                 summ+= a[i *N + k] * b[k*N + j];
            }
            c_cpu[i*N + j]=summ;
        }
    }
    unsigned int cpu_end = clock();
    double cpuTime = (double)(cpu_end - cpu_start);
    printf("time spent executing by the CPU: %.2f millseconds\n", cpuTime);

    delete a;
    delete b;
    delete c;
    return 0;
}
