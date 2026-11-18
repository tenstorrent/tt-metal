// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Bias-folded K-blocked walk (perf experiment height_bias_fold).
//
// The K-blocked walk with a bias used to pack fp32 partials into cb_matmul_partials with packer L1 accumulation over
// every K-block and then run a separate pass (add_bias_bcast_rows) that re-read every partial and the bias row, added
// them through SrcA / SrcB (the partial at tf32) and packed the output. This walk has no separate pass:
//  - K-block 0 seeds DEST with the bias, row-broadcast from cb_bias (unpacked through SrcB, copied B2D into the fp32
//    DEST: exact), before its matmul;
//  - K-blocks 1 .. n-2 accumulate onto the partials with packer L1 accumulation, as before;
//  - the last K-block reloads the fp32 partials into DEST (copy; exact when cb_matmul_partials unpacks to DEST),
//    runs its matmul on top and packs the output (ReLU on that pack only).
// One K-block: seed, matmul, pack the output (no partials at all).
// add_at_reload (fold_add): no seed; the last K-block's reload is the old bias pass's arithmetic (partials through SrcA
// + bias row-broadcast through SrcB, ELWADD into DEST), then its matmul on top. conditional_reconfig (fold_add_c): the
// reload's format switches touch only the source whose format changes.
//
// Measured (Blackhole p150b, LoFi, focus cases): every arrangement is slower than the separate pass. Each seed or
// reload costs one unpack/math init + format swap per output subblock (~150-260 cycles per 4-tile subblock), the
// separate pass streams at ~50 cycles per tile with one init per M-block, and the reload still reads every fp32
// partial. See the experiment's test_hbf.py for the variants.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block has no hook that writes DEST before the first K-block's
// matmul of each subblock (PostComputeFn fires only on the last K-block, PreKBlockFn once per K-block before DEST is
// acquired), so it cannot seed the bias. The K loop below is the helper's TileRowMajor + packer_l1_acc +
// LastBlockTarget::Out(WithRelu) path (ShortAfterPreKBlock init, WaitAndPopPerKBlock in0, per-subblock pack reconfig,
// same reload sequence) with the seed added; in1 can be retained on the last K-block (the face-direct stream).

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::bias_fold {

// DEST[r * w + c] <- bias tile (n0 + c), row 0 broadcast over the tile's rows; r < h, c < w. Restores the matmul.
template <uint32_t cb_bias, uint32_t cb_in1, uint32_t h, uint32_t w, uint32_t bk>
ALWI void seed_subblock(uint32_t cb_in0, uint32_t n0) {
    reconfig_data_format_srcb(cb_in0, cb_bias);
    unary_bcast_init<BroadcastType::ROW>(cb_bias);
    for (uint32_t r = 0; r < h; ++r) {
        for (uint32_t c = 0; c < w; ++c) {
            unary_bcast<BroadcastType::ROW>(cb_bias, n0 + c, r * w + c);
        }
    }
    reconfig_data_format_srcb(cb_bias, cb_in0);
    ckernel::matmul_block_init(cb_in0, cb_in1, false, w, h, bk);
}

// One M-block (or one in0 row group): num_in0_subblocks subblock rows x all bn columns, num_k_blocks K-blocks of bk.
// pre_k_block(block, num_k_blocks, is_last) runs before each K-block (the tilize); in0_source(block, cb_in0) picks the
// K-block's in0 CB (the center-tap copy). in1 is popped per K-block, except the last one when retain_in1_on_last.
// cb_bias must hold bn tiles fronted (waited here at first use).
template <
    uint32_t cb_in0,
    uint32_t cb_in1,
    uint32_t cb_out,
    uint32_t cb_interm,
    uint32_t cb_bias,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t bk,
    uint32_t bn,
    bool relu,
    bool retain_in1_on_last,
    bool add_at_reload,
    bool conditional_reconfig,
    typename PreKBlockFn,
    typename In0SourceFn>
ALWI void walk(uint32_t num_in0_subblocks, uint32_t num_k_blocks, PreKBlockFn pre_k_block, In0SourceFn in0_source) {
    static_assert(bn % sb_w == 0);
    constexpr uint32_t in1_num_subblocks = bn / sb_w;
    constexpr uint32_t row_group_tiles = sb_h * bn;
    constexpr uint32_t in1_block_tiles = bn * bk;
    const uint32_t in0_block_tiles = num_in0_subblocks * sb_h * bk;
    CircularBuffer in1(cb_in1);
    CircularBuffer out(cb_out);
    CircularBuffer interm(cb_interm);
    CircularBuffer bias(cb_bias);

    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        const bool first = block == 0;
        const bool last = block == num_k_blocks - 1;
        if constexpr (relu) {
            // As OutWithRelu: armed before the hook, which clears it around its own pack and re-arms it.
            if (last) {
                PACK((llk_pack_relu_config(ReluConfig::zero())));
            }
        }
        pre_k_block(block, num_k_blocks, last);
        const uint32_t a_in0 = in0_source(block, cb_in0);
        if (add_at_reload ? last : first) {
            // The bias is waited for at its first use (see the compute kernel's add_bias).
            MaybeDeviceZoneScope("compute_bias_wait");
            bias.wait_front(bn);
        }
        reconfig_data_format(cb_in1, a_in0);
        PACK((pack_reconfig_data_format(cb_interm)));
        ckernel::matmul_block_init(a_in0, cb_in1, false, sb_w, sb_h, bk);

        CircularBuffer in0(a_in0);
        in0.wait_front(in0_block_tiles);
        in1.wait_front(in1_block_tiles);
        const uint32_t target = last ? cb_out : cb_interm;

        uint32_t in0_offset = 0;
        for (uint32_t s0 = 0; s0 < num_in0_subblocks; ++s0) {
            if (last) {
                out.reserve_back(row_group_tiles);
            } else {
                interm.reserve_back(row_group_tiles);
            }
            for (uint32_t s1 = 0; s1 < in1_num_subblocks; ++s1) {
                tile_regs_acquire();
                if (first && (!add_at_reload || last)) {
                    seed_subblock<cb_bias, cb_in1, sb_h, sb_w, bk>(a_in0, s1 * sb_w);
                } else if (last && add_at_reload) {
                    // Reload with the bias: DEST <- partials (SrcA) + bias row 0 broadcast (SrcB), the old bias pass's
                    // arithmetic, then the last K-block's matmul on top.
                    if constexpr (conditional_reconfig) {
                        reconfig_data_format(cb_in1, cb_interm, a_in0, cb_bias);
                    } else {
                        reconfig_data_format(cb_interm, cb_bias);
                    }
                    add_bcast_rows_init(cb_interm, cb_bias);
                    if (s1 == 0) {
                        interm.wait_front(row_group_tiles);
                    }
                    for (uint32_t r = 0; r < sb_h; ++r) {
                        for (uint32_t c = 0; c < sb_w; ++c) {
                            add_tiles_bcast_rows(
                                cb_interm, cb_bias, r * bn + s1 * sb_w + c, s1 * sb_w + c, r * sb_w + c);
                        }
                    }
                    if (s1 == in1_num_subblocks - 1) {
                        interm.pop_front(row_group_tiles);
                    }
                    if constexpr (conditional_reconfig) {
                        reconfig_data_format(cb_interm, cb_in1, cb_bias, a_in0);
                    } else {
                        reconfig_data_format(cb_in1, a_in0);
                    }
                    ckernel::matmul_block_init(a_in0, cb_in1, false, sb_w, sb_h, bk);
                } else if (last) {
                    // The helper's reload: the partials' row group, this subblock's columns, into contiguous DEST.
                    reconfig_data_format_srca(cb_in1, cb_interm);
                    copy_init(cb_interm);
                    if (s1 == 0) {
                        interm.wait_front(row_group_tiles);
                    }
                    compute_kernel_lib::copy_subblock_row_strided(cb_interm, s1 * sb_w, bn, sb_h, sb_w);
                    if (s1 == in1_num_subblocks - 1) {
                        interm.pop_front(row_group_tiles);
                    }
                    reconfig_data_format_srca(cb_interm, cb_in1);
                    ckernel::matmul_block_init(a_in0, cb_in1, false, sb_w, sb_h, bk);
                }
                uint32_t in0_index = in0_offset;
                uint32_t in1_index = s1 * sb_w;
                for (uint32_t k = 0; k < bk; ++k) {
                    ckernel::matmul_block(a_in0, cb_in1, in0_index, in1_index, 0, false, sb_w, sb_h, bk);
                    ++in0_index;
                    in1_index += bn;
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((pack_reconfig_data_format(target)));
                PACK((llk_pack_reconfig_l1_acc((first || last) ? 0 : 1)));
                compute_kernel_lib::pack_subblock_row_strided(0, target, s1 * sb_w, bn, sb_h, sb_w);
                tile_regs_release();
            }
            if (last) {
                out.push_back(row_group_tiles);
            } else {
                interm.push_back(row_group_tiles);
            }
            in0_offset += sb_h * bk;
        }
        // Drain the L1-accumulated partials back to the CB base for the next K-block; the second-to-last K-block's
        // stay fronted for the last K-block's reload.
        if (num_k_blocks >= 2 && block < num_k_blocks - 2) {
            for (uint32_t s0 = 0; s0 < num_in0_subblocks; ++s0) {
                interm.wait_front(row_group_tiles);
                interm.pop_front(row_group_tiles);
            }
        }
        in0.pop_front(in0_block_tiles);
        if (!(retain_in1_on_last && last)) {
            in1.pop_front(in1_block_tiles);
        }
    }
}

}  // namespace conv2d_3x3::bias_fold
