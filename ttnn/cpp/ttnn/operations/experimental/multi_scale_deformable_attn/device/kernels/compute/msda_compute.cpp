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
// Reader contract:
//   * input_tile: only rows that are both in-range (r < v_rows) AND have an
//     in-bounds corner are written. Tail / OOB rows are left untouched
//     (stale CB bytes).
//   * scalar_tile: col 0 of TL/BL is explicitly written for all 32 rows,
//     with bf16 0 for tail / OOB-corner rows. Non-col-0 lanes are not
//     written.
// We rely on mul_tiles_bcast<COL>'s clear_fp32_dst_acc=true to zero DST so
// that only col-0 broadcasts contribute, and on scalar=0 to zero out the
// contribution of any stale input row.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t reduction_size = get_compile_time_arg_val(3);  // = 4 * P

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    CircularBuffer input_cb(input_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(input_cb_index, scalar_cb_index, output_cb_index);

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        // Reserve one output tile slot; we accumulate into it via L1 acc.
        output_cb.reserve_back(1);

        for (uint32_t i = 0; i < reduction_size; ++i) {
            if (i == 0) {
                pack_reconfig_l1_acc(0);  // first iter: overwrite L1
            } else if (i == 1) {
                pack_reconfig_l1_acc(1);  // subsequent: accumulate into L1
            }

            input_cb.wait_front(1);
            scalar_cb.wait_front(1);

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

            input_cb.pop_front(1);
            scalar_cb.pop_front(1);
        }

        // Reset L1-acc mode for the next output tile.
        pack_reconfig_l1_acc(0);
        output_cb.push_back(1);
    }
}
