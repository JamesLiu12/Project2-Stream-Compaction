/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <random>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/sort.h>
#include <stream_compaction/shared.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 22; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int SHARED_NAIVE_SIZE = 1 << 10;
const int SHARED_EFFICIENT_SIZE = 1 << 11;
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
int sharedInput[SHARED_EFFICIENT_SIZE];
int sharedExpected[SHARED_EFFICIENT_SIZE];
int sharedOutput[SHARED_EFFICIENT_SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");

    printf("******************************\n");
    printf("** SHARED MEMORY SCAN TESTS **\n");
    printf("******************************\n");

    // Shared memory scan tests

    genArray(SHARED_EFFICIENT_SIZE - 1, sharedInput, 50);
    sharedInput[SHARED_EFFICIENT_SIZE - 1] = 0;
    printArray(SHARED_EFFICIENT_SIZE, sharedInput, true);

    zeroArray(SHARED_EFFICIENT_SIZE, sharedExpected);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SHARED_EFFICIENT_SIZE, sharedExpected, sharedInput);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SHARED_EFFICIENT_SIZE, sharedExpected, true);

    zeroArray(SHARED_EFFICIENT_SIZE, sharedOutput);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(SHARED_EFFICIENT_SIZE - 3, sharedOutput, sharedInput);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SHARED_EFFICIENT_SIZE - 3, sharedOutput, true);
    printCmpResult(SHARED_EFFICIENT_SIZE - 3, sharedExpected, sharedOutput);

    zeroArray(SHARED_EFFICIENT_SIZE, sharedOutput);
    printDesc("shared naive scan, power-of-two");
    StreamCompaction::Shared::naiveScan(SHARED_NAIVE_SIZE, sharedOutput, sharedInput);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SHARED_NAIVE_SIZE, sharedOutput, true);
    printCmpResult(SHARED_NAIVE_SIZE, sharedExpected, sharedOutput);
    zeroArray(SHARED_EFFICIENT_SIZE, sharedOutput);
    printDesc("shared naive scan, non-power-of-two");
    StreamCompaction::Shared::naiveScan(SHARED_NAIVE_SIZE - 3, sharedOutput, sharedInput);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SHARED_NAIVE_SIZE - 3, sharedOutput, true);
    printCmpResult(SHARED_NAIVE_SIZE - 3, sharedExpected, sharedOutput);
    zeroArray(SHARED_EFFICIENT_SIZE, sharedOutput);
    printDesc("shared efficient scan, power-of-two");
    StreamCompaction::Shared::efficientScan(SHARED_EFFICIENT_SIZE, sharedOutput, sharedInput);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SHARED_EFFICIENT_SIZE, sharedOutput, true);
    printCmpResult(SHARED_EFFICIENT_SIZE, sharedExpected, sharedOutput);
    zeroArray(SHARED_EFFICIENT_SIZE, sharedOutput);
    printDesc("shared efficient scan, non-power-of-two");
    StreamCompaction::Shared::efficientScan(SHARED_EFFICIENT_SIZE - 3, sharedOutput, sharedInput);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SHARED_EFFICIENT_SIZE - 3, sharedOutput, true);
    printCmpResult(SHARED_EFFICIENT_SIZE - 3, sharedExpected, sharedOutput);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    // Radix Sort tests

    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> distribution(
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max());
    for (int i = 0; i < SIZE; i++) {
        a[i] = distribution(rng);
    }
    printArray(SIZE, a, true);

    printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two");
    StreamCompaction::Sort::radixSort(SIZE, c, a);
    printElapsedTime(StreamCompaction::Sort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, non-power-of-two");
    StreamCompaction::Sort::radixSort(NPOT, c, a);
    printElapsedTime(StreamCompaction::Sort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
