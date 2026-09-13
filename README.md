CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Sizhe Liu
  * [LinkedIn](https://www.linkedin.com/in/sizhe-liu-2726492b6/), [GithubGithub](https://github.com/JamesLiu12).
* Tested on: Windows 11 Pro, i9-12900H @ 2.50GHz 32GB, RTX 3090 24GB (Personal Computer)
* CUDA 13.3; Visual Studio 2022; Release build.

## Overview

A CUDA implementation of exclusive scan and stream compaction, with radix sort as an extra feature. The project compares a serial CPU scan with three GPU versions to see how the amount of work and memory access pattern affect performance.

- **CPU:** a simple loop computes the running sum.
- **Naive:** each step adds values from an increasing offset, using two buffers to keep reads and writes separate. This takes $O(N \log N)$ work.
- **Work-Efficient:** an upsweep and downsweep compute the scan with $O(N)$ work. Inputs are padded to the next power of two when needed.
- **Thrust:** `thrust::exclusive_scan` serves as a library baseline.

Stream compaction removes zeroes while keeping the remaining values in order. The GPU version builds flags, scans them to find output positions, and scatters the retained values.

## Performance Analysis

Each configuration was tested 10 times in Release mode. Plots show median times, with error bars marking the middle 50% of results. CPU timing uses `std::chrono`; GPU timing uses CUDA events and excludes initial allocation and input/output transfers.

| Experiment | Variable | Fixed settings |
|---|---|---|
| Block size | 32, 64, 128, 256, 512, 1024 | N = $2^{10}$, $2^{16}$, $2^{22}$; Naive and Work-Efficient |
| Array size | $2^{10}$ to $2^{24}$, increasing by 4× | Naive block size 256; Work-Efficient block size 64; CPU and Thrust baselines |

### 1. Effect of Block Size

![Scan runtime versus block size](images/performance/block_size.svg)

*Each panel uses a different vertical scale.*

The best block size changes with input size, and larger blocks are not always faster. At $2^{22}$ elements, Naive is fastest at 256 threads per block, while Work-Efficient is fastest at 64. These settings are used for the array-size comparison.

### 2. Effect of Array Size

![Scan runtime versus array size](images/performance/scan_size.svg)

*Both axes use a log scale.*

The CPU is much faster for small arrays. GPU times change relatively little at first, suggesting that launching kernels costs more than the small amount of work they perform. All three GPU versions first beat the CPU at $2^{22}$ among the tested sizes.

Work-Efficient is slower than Naive on small inputs despite doing less total work. Each tree level needs a separate kernel: at $2^{22}$ elements, it launches 43 kernels, compared with Naive's 23. The extra launches likely explain part of the difference. With larger arrays, the reduction in work becomes more useful, and Work-Efficient is about twice as fast as Naive at $2^{24}$.

![Scan speedup over CPU](images/performance/scan_speedup.svg)

*Speedup = CPU time / GPU time. Values above 1 mean the GPU version is faster.*

| Implementation | Time at $2^{24}$ (ms) | Speedup over CPU |
|---|---:|---:|
| CPU | 6.597 | 1.00× |
| Naive | 4.658 | 1.42× |
| Work-Efficient | 2.303 | 2.87× |
| Thrust | 0.847 | 7.79× |

Thrust performs best at large sizes. One noticeable result is its jump from about 0.127 ms at $2^{16}$ to 0.514 ms at $2^{18}$. A change in the library's execution strategy or setup cost could explain this, but the timing results alone do not show the cause.

### 3. Possible Bottlenecks

Nsight Systems helps show where time is spent between kernels, while Nsight Compute gives a closer look at one Naive scan step. These tests used $2^{24}$ elements and block size 128, which differs from the settings used in the performance graphs.

Naive seems to spend more effort moving data than doing calculations. Compute shows about 91.5% memory throughput but only 21–22% compute throughput for the tested kernel. This makes sense because every step reads and writes the whole array, even though each thread only performs a simple addition.

Work-Efficient cuts down the total work, but it still launches a separate kernel for every tree level. The Systems timeline shows noticeable gaps between these kernels. Near the root, some launches have only one block, so most of the GPU has no work to do. This helps explain why doing fewer additions does not always make it faster, especially for small arrays.

Thrust uses just two CUB kernels for this scan, compared with 25 for Naive and 47 for Work-Efficient. There is much less back-and-forth between kernel launches. Systems also shows temporary memory allocation and synchronization inside the Thrust call, so its total time includes more than just running the scan kernels.

## Extra Credit: Work-Efficient Scan

Each thread handles one active pair of tree nodes. At level `d`, only `N >> d` threads are needed, so the block count changes with each level. Extra threads return before computing their array indices. This avoids launching a full array of threads when most would have no work.

I also removed the separate root reset, `cudaMemset(dev_data + N - 1, 0, sizeof(int))`. The first downsweep uses zero as the root prefix directly. The final upsweep is skipped (`d < logn`) because it only computes the total sum that would be discarded. For `N > 1`, this saves one kernel launch and one memset; `N == 1` still needs a separate reset.

## Extra Credit: Radix Sort

The radix sort processes one bit at a time for 32 passes. Flipping the sign bit in the sorting key allows negative integers to be sorted correctly as well.

```cpp
StreamCompaction::Sort::radixSort(n, output, input);
```

For example, `[3, -2, 0, -7, 3]` becomes `[-7, -2, 0, 3, 3]`.

## Test Results

The tests cover power-of-two and non-power-of-two inputs, compaction, and radix sort with positive and negative integers.

<details>
<summary>Full test output: N = 2²², block size 64</summary>

```text
****************
** SCAN TESTS **
****************
    [   9  27  15  10  41   0  49  26  35   0   0  36  29 ...  29   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.7219ms    (std::chrono Measured)
    [   0   9  36  51  61 102 102 151 177 212 212 212 248 ... 102675936 102675965 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.7183ms    (std::chrono Measured)
    [   0   9  36  51  61 102 102 151 177 212 212 212 248 ... 102675860 102675902 ]
    passed 
==== naive scan, power-of-two ====
   elapsed time: 1.25811ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 1.17507ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.880192ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.944096ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.628832ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.701312ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   3   1   2   3   0   1   0   3   0   0   2   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 7.9118ms    (std::chrono Measured)
    [   1   3   1   2   3   1   3   2   1   3   3   1   2 ...   2   1 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 7.8696ms    (std::chrono Measured)
    [   1   3   1   2   3   1   3   2   1   3   3   1   2 ...   1   2 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 12.0065ms    (std::chrono Measured)
    [   1   3   1   2   3   1   3   2   1   3   3   1   2 ...   2   1 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 1.06195ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.906336ms    (CUDA Measured)
    passed 

**********************
** RADIX SORT TESTS **
**********************
    [ -901116820 1592269647 1036911988 501803631 -506353126 -1469150353 -407123097 -1725644006 -807619672 -1716690643 1583060036 842370173 -77060991 ... 601899222 -1338564959 ]
==== cpu sort, power-of-two ====
   elapsed time: 285.345ms    (std::chrono Measured)
    [ -2147483517 -2147480774 -2147480188 -2147479595 -2147476996 -2147475775 -2147474001 -2147473963 -2147473272 -2147472815 -2147468598 -2147465769 -2147465595 ... 2147483439 2147483532 ]
==== radix sort, power-of-two ====
   elapsed time: 35.7783ms    (CUDA Measured)
    [ -2147483517 -2147480774 -2147480188 -2147479595 -2147476996 -2147475775 -2147474001 -2147473963 -2147473272 -2147472815 -2147468598 -2147465769 -2147465595 ... 2147483439 2147483532 ]
    passed 
==== cpu sort, non-power-of-two ====
   elapsed time: 287.995ms    (std::chrono Measured)
    [ -2147483517 -2147480774 -2147480188 -2147479595 -2147476996 -2147475775 -2147474001 -2147473963 -2147473272 -2147472815 -2147468598 -2147465769 -2147465595 ... 2147483439 2147483532 ]
==== radix sort, non-power-of-two ====
   elapsed time: 39.541ms    (CUDA Measured)
Press any key to continue . . . 
    [ -2147483517 -2147480774 -2147480188 -2147479595 -2147476996 -2147475775 -2147474001 -2147473963 -2147473272 -2147472815 -2147468598 -2147465769 -2147465595 ... 2147483439 2147483532 ]
    passed
```

</details>
