// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel for fused multi-scale deformable attention. Used
// unchanged by both the V1 and the V2 reader — the readers differ only in how
// they produce the sample stream, never in what the stream means.
//
// Per output block (up to 32 queries of one (batch, head), packed vertically,
// spanning n_d_tiles tiles side by side for D > 32):
//
//   for each of REDUCTION_SIZE = 4 * L * P (input_tiles, scalar_tile) groups:
//     for each d-tile k:
//       dest[row, col] = input_tile_k[row, col] * scalar_tile[row, 0]   (COL bcast)
//       pack into output_cb slot k, accumulating in L1 after the first group
//
// The scalar carries `attention_weight * bilinear_corner_coefficient`, folded
// by the reader. So the four consecutive corner groups of a sampling point sum
// to the bilinear-interpolated, attention-weighted sample, and the L * P points
// then sum into the same accumulator. The accumulator is the op's output: no
// per-sample value is ever written back to DRAM, and no [B, Q, H, L, P, D]
// tensor exists at any point.
//
// The scalar tile is shared across all d-tiles of a group — the weight is per
// query row and independent of D.
//
// Reader contract (see fused_msda_reader_common.hpp):
//   * input_tile: only rows that are in range AND in bounds are written; other
//     rows hold stale CB bytes.
//   * scalar_tile: col 0 of TL/BL is written for all 32 rows, bf16 0 for tail
//     and out-of-bounds rows. Other lanes are not written.
// mul_tiles_bcast<COL> runs with clear_fp32_dst_acc=true, so DST is zeroed on
// entry and only col-0 broadcasts contribute; a 0 scalar then annihilates any
// stale input row.

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
constexpr uint32_t reduction_size = get_compile_time_arg_val(3);  // = 4 * L * P
constexpr uint32_t n_d_tiles = get_compile_time_arg_val(4);       // = ceil(D / 32)

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    CircularBuffer input_cb(input_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);

    compute_kernel_hw_startup(input_cb_index, scalar_cb_index, output_cb_index);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(input_cb_index, scalar_cb_index);

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        // Reserve the block's output tiles up front; the reduction accumulates
        // into them in L1, so they stay resident for all reduction_size groups.
        output_cb.reserve_back(n_d_tiles);

        for (uint32_t i = 0; i < reduction_size; ++i) {
            if (i == 0) {
                pack_reconfig_l1_acc(0);  // first group: overwrite
            } else if (i == 1) {
                pack_reconfig_l1_acc(1);  // subsequent groups: accumulate
            }

            input_cb.wait_front(n_d_tiles);
            scalar_cb.wait_front(1);

            for (uint32_t k = 0; k < n_d_tiles; ++k) {
                tile_regs_acquire();
                mul_tiles_bcast<BroadcastType::COL>(input_cb_index, scalar_cb_index, k, 0, 0);
                tile_regs_commit();

                tile_regs_wait();
                // out_of_order_output=true so each iteration packs to an
                // explicit slot (= k); the L1-acc mode then decides overwrite
                // vs accumulate. The default would auto-advance the write
                // pointer and clobber L1 past the block after the first group.
                pack_tile<true>(0, output_cb_index, k);
                tile_regs_release();
            }

            input_cb.pop_front(n_d_tiles);
            scalar_cb.pop_front(1);
        }

        pack_reconfig_l1_acc(0);
        output_cb.push_back(n_d_tiles);
    }
}
