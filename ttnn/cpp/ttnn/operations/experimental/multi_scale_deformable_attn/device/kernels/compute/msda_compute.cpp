// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for fused multi-scale deformable attention.
//
// Per output block (up to 32 queries packed vertically, one per row,
// spanning n_d_tiles tiles side by side for D > 32):
//   For each of REDUCTION_SIZE (= 4 * P) (input_tiles, scalar_tile) groups:
//     for each d-tile k:
//       dest[h, w] = input_tile_k[h, w] * scalar_tile[h, 0]   (COL broadcast)
//       pack into output_cb slot k with L1 accumulate (after first group)
// The scalar tile is shared across all d-tiles of a group: the combined
// (attn * bilinear) weight is per query row, independent of D.
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
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/compute/msda_geometry.hpp"

constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t input_rm_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t reduction_size = get_compile_time_arg_val(4);  // = 4 * P
constexpr uint32_t n_d_tiles = get_compile_time_arg_val(5);       // = ceil(D / 32)

// Geometry phase: the reader stages gx/gy/attn as column-0 tiles here, and takes
// back the floored corner per axis. Scale and shift carry the align_corners
// variant as fp32 bit patterns, so this kernel has no branch on it.
constexpr uint32_t grid_x_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t grid_y_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t attn_tile_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t x0_cb_index = get_compile_time_arg_val(9);
constexpr uint32_t y0_cb_index = get_compile_time_arg_val(10);
constexpr uint32_t frac_x_cb_index = get_compile_time_arg_val(11);
constexpr uint32_t frac_y_cb_index = get_compile_time_arg_val(12);
constexpr uint32_t P = get_compile_time_arg_val(13);
constexpr uint32_t x_scale_bits = get_compile_time_arg_val(14);
constexpr uint32_t x_shift_bits = get_compile_time_arg_val(15);
constexpr uint32_t y_scale_bits = get_compile_time_arg_val(16);
constexpr uint32_t y_shift_bits = get_compile_time_arg_val(17);

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    CircularBuffer input_cb(input_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);

    compute_kernel_hw_startup(input_cb_index, scalar_cb_index, output_cb_index);

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        // ---- GEOMETRY ----
        // Every point of the block is solved before the reduction starts. The
        // reader is holding all P grid tiles and cannot take a corner back until
        // it has pushed them, so interleaving the two phases would deadlock.
        init_sfpu(grid_x_cb_index, x0_cb_index);
        uint32_t last_srca = grid_x_cb_index;
        for (uint32_t p = 0; p < P; ++p) {
            msda_geometry::point(
                grid_x_cb_index,
                grid_y_cb_index,
                attn_tile_cb_index,
                x0_cb_index,
                y0_cb_index,
                frac_x_cb_index,
                frac_y_cb_index,
                scalar_cb_index,
                x_scale_bits,
                x_shift_bits,
                y_scale_bits,
                y_shift_bits,
                last_srca);
        }

        // ---- REDUCTION ----
        // Both inits run per group: the unpacker alternates between tilize and the broadcast
        // multiply, so neither configuration survives the other.
        // Reserve the block's output tiles; we accumulate into them via L1 acc.
        output_cb.reserve_back(n_d_tiles);

        for (uint32_t i = 0; i < reduction_size; ++i) {
            // Accumulation is off for the tilize below: it packs into input_cb, and with L1 acc
            // still armed from the previous group it would add to that buffer instead of
            // replacing it. Re-armed once the tilize has packed, for the reduction's own pack.
            pack_reconfig_l1_acc(0);

            // The reader stages the corner's sticks row-major; the unpacker tilizes them here.
            // Doing that placement on the reader's core would be a scalar loop against hardware
            // built for it, and on Blackhole a 32-byte face half is not even a legal DRAM read.
            // One block: 32 query rows is exactly one tile tall, n_d_tiles wide. The helper owns
            // the buffer handshake and the register reconfiguration either side of it.
            compute_kernel_lib::tilize<n_d_tiles, input_rm_cb_index, input_cb_index>(1);

            if (i > 0) {
                pack_reconfig_l1_acc(1);  // groups after the first accumulate into the output
            }
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(input_cb_index, scalar_cb_index);
            input_cb.wait_front(n_d_tiles);
            scalar_cb.wait_front(1);

            for (uint32_t k = 0; k < n_d_tiles; ++k) {
                tile_regs_acquire();
                mul_tiles_bcast<BroadcastType::COL>(input_cb_index, scalar_cb_index, k, 0, 0);
                tile_regs_commit();

                tile_regs_wait();
                // out_of_order_output=true so every iteration packs to an
                // explicit tile slot (= k). The L1-acc mode
                // (pack_reconfig_l1_acc) then decides between overwrite and
                // accumulate. Default (=false) would auto-advance the write
                // pointer and clobber out-of-range L1 after the first group.
                pack_tile<true>(0, output_cb_index, k);
                tile_regs_release();
            }

            input_cb.pop_front(n_d_tiles);
            scalar_cb.pop_front(1);
        }

        // Reset L1-acc mode for the next output block.
        pack_reconfig_l1_acc(0);
        output_cb.push_back(n_d_tiles);
    }
}
