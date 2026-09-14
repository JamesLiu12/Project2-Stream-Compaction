#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Shared {
        StreamCompaction::Common::PerformanceTimer& timer();

        void naiveScan(int n, int *odata, const int *idata);

        void efficientScan(int n, int* odata, const int* idata);
    }
}
