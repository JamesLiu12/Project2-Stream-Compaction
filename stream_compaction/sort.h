#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Sort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radixSort(int n, int* odata, const int* idata);
    }
}
