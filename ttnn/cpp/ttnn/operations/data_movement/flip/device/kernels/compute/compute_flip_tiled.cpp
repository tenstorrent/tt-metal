// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// flip is applied by the reader, this just passes tiles through FPU pipeline.

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    CircularBuffer cb_in_exp(cb_in);
    CircularBuffer cb_out_exp(cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_in_exp.wait_front(1);
        cb_out_exp.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_out_exp.push_back(1);
        cb_in_exp.pop_front(1);
    }
}
