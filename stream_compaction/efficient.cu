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
            int t = blockIdx.x * blockDim.x + threadIdx.x;

            if (t >= (n >> d)) return;

            int index = ((t + 1) << d) - 1;

            data[index] += data[index - (1 << (d - 1))];
        }

        __global__ void kernDownSweep(int n, int d, int *data)
        {
            int t = blockIdx.x * blockDim.x + threadIdx.x;

            if (t >= (n >> d)) return;

            int index = ((t + 1) << d) - 1;
            int leftIndex = index - (1 << (d - 1));

            int right = 1 << d == n ? 0 : data[index];

            data[index] = right + data[leftIndex];
            data[leftIndex] = right;
        }

        __global__ void kernScatter(int n, int *odata, int* idata, int *indicies)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (idata[index] != 0) {
                odata[indicies[index]] = idata[index];
            }
        }

        void scanDevice(int N, int logn, int *dev_data)
        {
            constexpr int blockSize = 128;

            if (N == 1) {
                cudaMemset(dev_data, 0, sizeof(int));
                return;
            }

            for (int d = 1; d < logn; d++) {
                dim3 blocksPerGrid(((N >> d) + blockSize - 1) / blockSize);
                dim3 threadsPerBlock(blockSize);

                kernUpSweep<<<blocksPerGrid, threadsPerBlock>>>(N, d, dev_data);
                checkCUDAError("kernUpSweep");
            }

            for (int d = logn; d >= 1; d--) {
                dim3 blocksPerGrid(((N >> d) + blockSize - 1) / blockSize);
                dim3 threadsPerBlock(blockSize);

                kernDownSweep<<<blocksPerGrid, threadsPerBlock>>>(N, d, dev_data);
                checkCUDAError("kernDownSweep");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int* dev_data;

            int logn = ilog2ceil(n);
            int N = 1 << logn;

            cudaMalloc(&dev_data, sizeof(int) * N);
            cudaMemset(dev_data, 0, sizeof(int) * N);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanDevice(N, logn, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
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
            if (n <= 0) {
                return 0;
            }

            int* dev_indices;
            int* dev_odata;
            int* dev_idata;

            int logn = ilog2ceil(n);
            int N = 1 << logn;

            cudaMalloc(&dev_indices, sizeof(int) * N);
            cudaMalloc(&dev_odata, sizeof(int) * n);
            cudaMalloc(&dev_idata, sizeof(int) * n);

            cudaMemset(dev_indices, 0, sizeof(int) * N);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            constexpr int blockSize = 128;

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, dev_indices, dev_idata);

            scanDevice(N, logn, dev_indices);

            kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_indices);

            timer().endGpuTimer();

            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += (idata[n - 1] != 0);

            cudaMemcpy(odata, dev_odata, sizeof(int) * count, cudaMemcpyDeviceToHost);

            cudaFree(dev_indices);
            cudaFree(dev_odata);
            cudaFree(dev_idata);

            return count;
        }
    }
}
