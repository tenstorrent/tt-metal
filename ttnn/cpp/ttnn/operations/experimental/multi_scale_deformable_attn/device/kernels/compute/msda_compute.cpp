// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for fused multi-scale deformable attention.
//
// Per output tile (up to 32 queries packed vertically, one per row):
//   For each of REDUCTION_SIZE (= 4 * P) (input_tile, scalar_tile) pairs:
//     dest[h, w] = input_tile[h, w] * scalar_tile[h, 0]   (COL broadcast)
//     pack into output_cb with L1 accumulate (after first iter)
//
// Reader fills all 32 rows of input_tile with per-query value sticks and
// writes per-query scalars into col 0 of TL/BL faces. Tail rows (queries
// past v_rows for the last tile in a sub-batch) are zero-filled by the
// reader so they accumulate to zero.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t reduction_size = get_compile_time_arg_val(3);  // = 4 * P

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(input_cb_index, scalar_cb_index, output_cb_index);

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        // Reserve one output tile slot; we accumulate into it via L1 acc.
        cb_reserve_back(output_cb_index, 1);

        for (uint32_t i = 0; i < reduction_size; ++i) {
            if (i == 0) {
                pack_reconfig_l1_acc(0);  // first iter: overwrite L1
            } else if (i == 1) {
                pack_reconfig_l1_acc(1);  // subsequent: accumulate into L1
            }

            cb_wait_front(input_cb_index, 1);
            cb_wait_front(scalar_cb_index, 1);

            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::COL>(input_cb_index, scalar_cb_index, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            // out_of_order_output=true so every iteration packs to the same
            // tile slot (= 0). The L1-acc mode (pack_reconfig_l1_acc) then
            // decides between overwrite and accumulate. Default (=false)
            // would auto-advance the write pointer and clobber out-of-range
            // L1 after iter 1.
            pack_tile<true>(0, output_cb_index, 0);
            tile_regs_release();

            cb_pop_front(input_cb_index, 1);
            cb_pop_front(scalar_cb_index, 1);
        }

        // Reset L1-acc mode for the next output tile.
        pack_reconfig_l1_acc(0);
        cb_push_back(output_cb_index, 1);
    }
}
