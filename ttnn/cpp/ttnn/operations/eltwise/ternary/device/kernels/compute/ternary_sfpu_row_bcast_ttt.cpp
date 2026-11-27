// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast
// Supports broadcasting 0, 1 or 2 inputs (A/B/C)
// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag
//
// The actual ternary operation is provided via:
//   TERNARY_SFPU_OP_INIT()
//   TERNARY_SFPU_OP_FUNC(src_idx_a, src_idx_b, src_idx_c, dst_idx_out)
// which are configured by the program factory for the desired op (e.g., where).

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/mac.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// A row-broadcast operand is expanded by unary_bcast straight into its DEST slot: packing the expanded tile to
// L1 and copying it back would round NaN to inf and -0.0 to +0.0 in a 16-bit DEST.
template <bool is_bcast>
ALWI void load_operand(const uint32_t dfb_id, const uint32_t dst_index) {
    if constexpr (is_bcast) {
        reconfig_data_format(dfb_id, dfb_id);
        unary_bcast_init<BroadcastType::ROW>(dfb_id);
        unary_bcast<BroadcastType::ROW>(dfb_id, 0, dst_index);
    } else {
        copy_init(dfb_id);
        copy_tile(dfb_id, 0, dst_index);
    }
}

void kernel_main() {
    compute_kernel_hw_startup(dfb::in_a, dfb::out);

    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // typically 1

    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    DataflowBuffer dfb_c(dfb::in_c);

    copy_init(dfb::in_a);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_out.reserve_back(num_tiles_per_cycle);

        dfb_a.wait_front(num_tiles_per_cycle);
        dfb_b.wait_front(num_tiles_per_cycle);
        dfb_c.wait_front(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load A -> DST[0], B -> DST[1], C -> DST[2]
        load_operand<BCAST_A != 0>(dfb::in_a, 0);
        load_operand<BCAST_B != 0>(dfb::in_b, 1);
        load_operand<BCAST_C != 0>(dfb::in_c, 2);

        // Execute configured ternary SFPU op
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Write out result
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        dfb_a.pop_front(num_tiles_per_cycle);
        dfb_b.pop_front(num_tiles_per_cycle);
        dfb_c.pop_front(num_tiles_per_cycle);
    }
}
