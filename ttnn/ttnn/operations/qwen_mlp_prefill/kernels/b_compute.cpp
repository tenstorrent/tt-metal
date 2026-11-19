// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"

void kernel_main() {
    constexpr uint32_t k_tiles = CT_ARG("i_tiles");
    constexpr uint32_t columns = CT_ARG("b_tiles_per_core");
    constexpr uint32_t chips = CT_ARG("chips");
    constexpr uint32_t lag = CT_ARG("owner_lag");
    constexpr uint32_t rows = CT_ARG("rows");
    constexpr uint32_t cb_a = CT_ARG("cb_a");
    constexpr uint32_t cb_wd = CT_ARG("cb_wd");
    constexpr uint32_t cb_own = CT_ARG("cb_partial_own");
    constexpr uint32_t cb_h = CT_ARG("cb_h");
    constexpr uint32_t cb_y = CT_ARG("cb_y");
    const uint32_t chip = RT_ARG("chip");
    CircularBuffer a(cb_a);
    CircularBuffer weights(cb_wd);
    CircularBuffer own(cb_own);
    CircularBuffer h(cb_h);
    CircularBuffer y(cb_y);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_wd, cb_own);
    matmul_block_init(cb_a, cb_wd, false, columns, 1, k_tiles);
    weights.wait_front(k_tiles * columns);

    for (uint32_t step = 0; step < rows + lag; step++) {
        if (step < rows) {
            a.wait_front(k_tiles);
            own.reserve_back(columns);
            tile_regs_acquire();
            for (uint32_t k = 0; k < k_tiles; k++) {
                matmul_block(cb_a, cb_wd, k, k * columns, 0, false, columns, 1, k_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < columns; i++) {
                pack_tile(i, cb_own);
            }
            tile_regs_release();
            own.push_back(columns);
            a.pop_front(k_tiles);
        }
        if (step >= lag) {
            own.wait_front(columns);
            h.wait_front(columns);
            y.reserve_back(columns);
            reconfig_data_format_srca(cb_own);
            copy_init(cb_own);
            tile_regs_acquire();
            for (uint32_t i = 0; i < columns; i++) {
                copy_tile(cb_own, i, i);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_h);
            for (uint32_t i = 0; i < columns; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_h, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < columns; i++) {
                pack_tile(i, cb_y);
            }
            tile_regs_release();
            y.push_back(columns);
            own.pop_front(columns);
            h.pop_front(columns);
            reconfig_data_format(cb_wd, cb_a);
            matmul_block_init(cb_a, cb_wd, false, columns, 1, k_tiles);
        }
    }
}
