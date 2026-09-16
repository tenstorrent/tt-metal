// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TILE-layout compute for qkv_causal_conv1d_silu.
//
// Same four-tap accumulation as the ROW_MAJOR kernel, byte for byte: each tap multiplies a shifted
// activation block by its per-channel tap tile (mul_tiles_bcast_rows), the running sum travels through the
// bf16 `partial` DFB, and the final tap applies SiLU before packing. The only difference is where the
// shifted block comes from. The ROW_MAJOR kernel tilizes a block the reader gathered stick by stick; here
// the reader hands over whole tiles (the current tile-row and the preceding one) and the shift is a matmul
// against constant 0/1 matrices:
//
//   shifted_d = S_cur[d] @ x_cur + S_prev[d] @ x_prev          (S_hist[d] instead of S_prev[d] at mt == 0)
//
// with d = 3 - tap the number of rows the tap looks back. d == 0 needs no shift at all: tap 3 reads the
// current tiles straight out of the reader's DFB. Every S entry is exactly 0.0 or 1.0 and every output
// element is one product 1.0 * x plus 31 exact zeros, so a shifted tile is a bit-exact permutation of the
// bf16 activation and the downstream arithmetic is unchanged.

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <uint32_t block_ct, uint32_t num_blocks>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    constexpr uint32_t shift_tile_count = 9;
    compute_kernel_hw_startup(dfb::act_tile, dfb::shift, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer shift(dfb::shift);
    DataflowBuffer shifted(dfb::shifted);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);
    silu_tile_init();

    shift.wait_front(shift_tile_count);
    if constexpr (num_blocks == 1) {
        weights.wait_front(tap_count * block_ct);
    }
    for (uint32_t item = 0; item < wi_count; ++item) {
        if constexpr (num_blocks > 1) {
            weights.wait_front(tap_count * block_ct);
        }
        // The first tile-row has no predecessor: its left context is the history tile-row, whose three carry
        // rows sit at tile rows 0-2 rather than 29-31, so it needs the S_hist matrices.
        const bool first_tile_row = (wi_start + item) / num_blocks == 0;
        activation.wait_front(2 * block_ct);

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            const uint32_t back = tap_count - 1 - tap;  // rows this tap looks back: 3, 2, 1, 0
            // Tap 3 reads the unshifted current tiles in place; the others consume a matmul-shifted block.
            uint32_t source_dfb = dfb::act_tile;
            uint32_t source_base = block_ct;
            if (back != 0) {
                const uint32_t cur_shift = back - 1;
                const uint32_t prev_shift = (first_tile_row ? 6u : 3u) + back - 1;
                // matmul_tiles(a, b): a -> srcB, b -> srcA, so reconfig takes (srcA, srcB) = (b, a).
                reconfig_data_format(dfb::act_tile, dfb::shift);
                matmul_init(dfb::shift, dfb::act_tile, 0);
                for (uint32_t ct = 0; ct < block_ct; ++ct) {
                    shifted.reserve_back(1);
                    tile_regs_acquire();
                    // DST starts cleared, and matmul_tiles accumulates into it.
                    matmul_tiles(dfb::shift, dfb::act_tile, cur_shift, block_ct + ct, 0);
                    matmul_tiles(dfb::shift, dfb::act_tile, prev_shift, ct, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, dfb::shifted);
                    shifted.push_back(1);
                    tile_regs_release();
                }
                shifted.wait_front(block_ct);
                source_dfb = dfb::shifted;
                source_base = 0;
            }

            const bool is_final_tap = tap + 1 == tap_count;
            const uint32_t destination_dfb = is_final_tap ? dfb::output : dfb::partial;
            if (tap != 0) {
                partial.wait_front(block_ct);
            }

            reconfig_data_format_srca(source_dfb);
            reconfig_data_format_srcb(dfb::weights);
            mul_bcast_rows_init(source_dfb, dfb::weights);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                if (is_final_tap) {
                    output.reserve_back(1);
                } else {
                    partial.reserve_back(1);
                }
                tile_regs_acquire();
                if (tap != 0) {
                    mul_bcast_rows_init(source_dfb, dfb::weights);
                }
                mul_tiles_bcast_rows(source_dfb, dfb::weights, source_base + ct, tap * block_ct + ct, 0);

                if (tap != 0) {
                    reconfig_data_format_srca(dfb::partial);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, 0, 0);
                    // The partial add binds srcA to the accumulator; restore activation for the next multiply.
                    reconfig_data_format_srca(source_dfb);
                }
                if (is_final_tap) {
                    silu_tile(0);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, destination_dfb);
                if (is_final_tap) {
                    output.push_back(1);
                } else {
                    partial.push_back(1);
                }
                if (tap != 0) {
                    partial.pop_front(1);
                }
                tile_regs_release();
            }
            if (back != 0) {
                shifted.pop_front(block_ct);
            }
        }
        activation.pop_front(2 * block_ct);
        if constexpr (num_blocks > 1) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
