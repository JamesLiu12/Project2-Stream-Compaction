#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

namespace StreamCompaction {
    namespace Shared {
        using Common::blockSize;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        constexpr int maxThreads = 1024;
        constexpr int logMaxThreads = 10;

        __global__ void kernNaiveScan(int n, int *odata, const int *idata)
        {
            __shared__ int shared[maxThreads << 1];

            int index = threadIdx.x;
            int* data1 = shared;
            int* data2 = shared + maxThreads;

            shared[index] = index < n ? idata[index] : 0;

            __syncthreads();

            for (int d = 0; (1 << d) < maxThreads; d++) {
                data2[index] = index >= (1 << d)
                    ? data1[index] + data1[index - (1 << d)]
                    : data1[index];

                __syncthreads();

                int* tmp = data1;
                data1 = data2;
                data2 = tmp;
            }

            if (index < n) {
                odata[index] = index ? data1[index - 1] : 0;
            }
        }

        __global__ void kernEfficientScan(int n, int* odata, const int* idata)
        {
            __shared__ int shared[maxThreads << 1];

            int t = threadIdx.x;
            int N = maxThreads << 1;
            int logn = logMaxThreads + 1;

            shared[t] = t < n ? idata[t] : 0;
            shared[t + maxThreads] = t + maxThreads < n
                ? idata[t + maxThreads]
                : 0;

            __syncthreads();

            for (int d = 1; d < logn; d++) {
                if (t < (N >> d)) {
                    int index = ((t + 1) << d) - 1;

                    shared[index] += shared[index - (1 << (d - 1))];
                }

                __syncthreads();
            }

            for (int d = logn; d >= 1; d--) {
                if (t < (N >> d)) {
                    int index = ((t + 1) << d) - 1;
                    int leftIndex = index - (1 << (d - 1));

                    int right = (1 << d) == N ? 0 : shared[index];

                    shared[index] = right + shared[leftIndex];
                    shared[leftIndex] = right;
                }

                __syncthreads();
            }

            if (t < n) {
                odata[t] = shared[t];
            }

            if (t + maxThreads < n) {
                odata[t + maxThreads] = shared[t + maxThreads];
            }
        }

        void naiveScan(int n, int *odata, const int *idata)
        {

            if (n <= 0 || n > maxThreads) {
                return;
            }

            int* dev_odata;
            int* dev_idata;

            cudaMalloc(&dev_odata, sizeof(int) * n);

            cudaMalloc(&dev_idata, sizeof(int) * n);

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            kernNaiveScan<<<1, maxThreads>>>(n, dev_odata, dev_idata);
            checkCUDAError("kernNaiveScan");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }

        void efficientScan(int n, int* odata, const int* idata)
        {
            if (n <= 0 || n > (maxThreads << 1)) {
                return;
            }

            int* dev_odata;
            int* dev_idata;

            cudaMalloc(&dev_odata, sizeof(int) * n);
            cudaMalloc(&dev_idata, sizeof(int) * n);

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            kernEfficientScan<<<1, maxThreads>>>(n, dev_odata, dev_idata);
            checkCUDAError("kernEfficientScan");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}