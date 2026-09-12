#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using Common::blockSize;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernStep(int n, int offset, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }


            odata[index] = idata[index];
            if (index >= offset)
            {
                odata[index] += idata[index - offset];
            }
        }

        __global__ void kernToExclusive(int n, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) return;

            odata[index] = index == 0 ? 0 : idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int* dev_idata;
            int* dev_odata;

            cudaMalloc(&dev_idata, sizeof(int) * n);
            cudaMalloc(&dev_odata, sizeof(int) * n);

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            const dim3 blocksPerGrid = (n + blockSize - 1) / blockSize;
            const dim3 threadsPerBlock = dim3(blockSize);

            const int logn = ilog2ceil(n);

            for (int d = 0; d < logn; d++)
            {
                const int offset = 1 << d;

                kernStep<<<blocksPerGrid, threadsPerBlock>>>(n, offset, dev_odata, dev_idata);
                checkCUDAError("kernStep");

                std::swap(dev_odata, dev_idata);
            }

            kernToExclusive<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);
            checkCUDAError("kernToExclusive");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
