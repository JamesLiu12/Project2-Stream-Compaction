#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int *data)
        {
            int index = ((blockIdx.x * blockDim.x + threadIdx.x + 1) << d) - 1;

            if (index >= n) {
                return;
            }

            data[index] += data[index - (1 << (d - 1))];
        }

        __global__ void kernDownSweep(int n, int d, int* data)
        {
            int index = ((blockIdx.x * blockDim.x + threadIdx.x + 1) << d) - 1;

            if (index >= n) {
                return;
            }

            int leftIndex = index - (1 << (d - 1));

            int tmpLeft = data[leftIndex];
            data[leftIndex] = data[index];
            data[index] += tmpLeft;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int* dev_data;

            int logn = ilog2ceil(n);
            int N = 1 << logn;

            cudaMalloc(&dev_data, sizeof(int) * N);
            cudaMemset(dev_data, 0, sizeof(int) * N);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            constexpr int blockSize = 128;

            for (int d = 1; d <= logn; d++) {
                dim3 blocksPerGrid(((N >> d) + blockSize - 1) / blockSize);
                dim3 threadsPerBlock(blockSize);

                kernUpSweep<<<blocksPerGrid, threadsPerBlock>>>(N, d, dev_data);
                checkCUDAError("kernUpSweep");
            }

            cudaMemset(dev_data + N - 1, 0, sizeof(int));

            for (int d = logn; d >= 1; d--) {
                dim3 blocksPerGrid(((N >> d) + blockSize - 1) / blockSize);
                dim3 threadsPerBlock(blockSize);

                kernDownSweep<<<blocksPerGrid, threadsPerBlock>>>(N, d, dev_data);
                checkCUDAError("kernDownSweep");
            }

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
