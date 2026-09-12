#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sort.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Sort {
        using Common::blockSize;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernComputeCurrentBits(int n, int d, int *currentBits, int *idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            unsigned int key = static_cast<unsigned int>(idata[index]) ^ 0x80000000u;

            currentBits[index] = (key >> d) & 1u;
        }

        __global__ void kernFlipBools(int n, int *obits, int *ibits)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            obits[index] = ibits[index] ^ 1;
        }

        __global__ void kernComputeOneOffsets(int n, int *oneOffsets, int *zeroOffsets, int *zeroFlags)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }
            
            oneOffsets[index] = index - zeroOffsets[index] + zeroOffsets[n - 1] + zeroFlags[n - 1];
        }

        __global__ void kernComputeDstIndices(int n, int *dstIndices, int* oneOffsets, int *zeroOffsets, int *currentBits)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            dstIndices[index] = currentBits[index] ? oneOffsets[index] : zeroOffsets[index];
        }

        __global__ void kernComputeOutput(int n, int *odata, int *idata, int *indices)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            odata[indices[index]] = idata[index];
        }

        static void split(
            int n, int d, 
            int* dev_odata, 
            int* dev_idata, 
            int* dev_currentBits, 
            int* dev_zeroFlags, 
            int* dev_zeroOffsets, 
            int* dev_oneOffsets, 
            int* dev_dstIndices)
        {
            int logn = ilog2ceil(n);
            int N = 1 << logn;

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 threadsPerBlock(blockSize);

            kernComputeCurrentBits<<<blocksPerGrid, threadsPerBlock>>>(n, d, dev_currentBits, dev_idata);
            kernFlipBools<<<blocksPerGrid, threadsPerBlock>>>(n, dev_zeroFlags, dev_currentBits);

            cudaMemcpy(dev_zeroOffsets, dev_zeroFlags, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            if (N > n) {
                cudaMemset(dev_zeroOffsets + n, 0, sizeof(int) * (N - n));
            }

            Efficient::scanDevice(N, logn, dev_zeroOffsets);

            kernComputeOneOffsets<<<blocksPerGrid, threadsPerBlock>>>(n, dev_oneOffsets, dev_zeroOffsets, dev_zeroFlags);
            kernComputeDstIndices<<<blocksPerGrid, threadsPerBlock>>>(n, dev_dstIndices, dev_oneOffsets, dev_zeroOffsets, dev_currentBits);
            kernComputeOutput<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_dstIndices);
        }

        void radixSort(int n, int *odata, const int *idata)
        {
            if (n <= 0) {
                return;
            }

            int* dev_odata;
            int* dev_idata;
            int* dev_currentBits;
            int* dev_zeroFlags;
            int* dev_zeroOffsets;
            int* dev_oneOffsets;
            int* dev_dstIndices;

            int logn = ilog2ceil(n);
            int N = 1 << logn;

            cudaMalloc(&dev_odata, sizeof(int) * n);
            cudaMalloc(&dev_idata, sizeof(int) * n);
            cudaMalloc(&dev_currentBits, sizeof(int) * n);
            cudaMalloc(&dev_zeroFlags, sizeof(int) * n);
            cudaMalloc(&dev_zeroOffsets, sizeof(int) * N);
            cudaMalloc(&dev_oneOffsets, sizeof(int) * n);
            cudaMalloc(&dev_dstIndices, sizeof(int) * n);

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d < 32; d++)
            {
                split(
                    n, d,
                    dev_odata,
                    dev_idata,
                    dev_currentBits,
                    dev_zeroFlags,
                    dev_zeroOffsets,
                    dev_oneOffsets,
                    dev_dstIndices);

                std::swap(dev_idata, dev_odata);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_currentBits);
            cudaFree(dev_zeroFlags);
            cudaFree(dev_zeroOffsets);
            cudaFree(dev_oneOffsets);
            cudaFree(dev_dstIndices);
        }
    }
}