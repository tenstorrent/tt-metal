// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SCAFFOLDING compute for the gated-delta prefill-then-query op.
// Placeholder: copies cb_in -> cb_state_out (an exact state passthrough) and
// cb_in -> cb_o_out (placeholder output token). The real kernel will implement
// the gated delta-rule recurrence over the K/V sequence plus the final query.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_state_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_o_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_o_out = tt::CBIndex::c_17;

    compute_kernel_hw_startup(cb_in, cb_state_out);
    copy_tile_to_dst_init_short(cb_in);

    CircularBuffer cb_in_o(cb_in);
    CircularBuffer cb_state_out_o(cb_state_out);
    CircularBuffer cb_o_out_o(cb_o_out);

    // State passthrough (fp32 -> fp32).
    for (uint32_t t = 0; t < num_state_tiles; t++) {
        cb_in_o.wait_front(1);
        cb_state_out_o.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_state_out, 0);
        tile_regs_release();
        cb_state_out_o.push_back(1);
        cb_in_o.pop_front(1);
    }

    // Placeholder output token (fp32 -> bf16).
    for (uint32_t t = 0; t < num_o_tiles; t++) {
        cb_in_o.wait_front(1);
        cb_o_out_o.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o_out, 0);
        tile_regs_release();
        cb_o_out_o.push_back(1);
        cb_in_o.pop_front(1);
    }
}
