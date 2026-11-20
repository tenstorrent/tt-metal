// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rect.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t row_tiles = CT_ARG("n_row_tiles");
    constexpr uint32_t depth = CT_ARG("n_depth");
    constexpr uint32_t norm_cores = CT_ARG("norm_cores");
    constexpr uint32_t rows = CT_ARG("rows");
    const Rect norm[norm_cores] = {RECT("norm0"), RECT("norm1")};
    CircularBuffer n(CT_ARG("cb_n"));

    uint32_t credited = 0;
    uint32_t received = 0;
    while (received < rows) {
        if (credited < rows && n.pages_reservable_at_back((credited - received + 1) * row_tiles)) {
            const uint32_t producer = credited % norm_cores;
            Semaphore<>(CT_ARG("sem_n_free") + credited % depth)
                .up(noc, norm[producer].x_start, norm[producer].y_start, 1);
            credited++;
        }
        if (Semaphore<>(CT_ARG("sem_n_valid") + received % depth).value() > received / depth) {
            n.push_back(row_tiles);
            received++;
        }
    }
    noc.async_atomic_barrier();
}
