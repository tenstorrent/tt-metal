// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    const Rect norm_grid = RECT("norm_grid");
    CircularBuffer n(CT_ARG("cb_n"));
    constexpr uint32_t depth = CT_ARG("n_depth");

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        const uint32_t slot = m % depth;
        n.reserve_back(row_tiles);
        Semaphore<> n_free(CT_ARG("sem_n_free") + slot);
        norm_grid.signal(noc, n_free, 1);
        Semaphore<>(CT_ARG("sem_n_valid") + slot).wait_min(m / depth + 1);

        n.push_back(row_tiles);
    }
}
