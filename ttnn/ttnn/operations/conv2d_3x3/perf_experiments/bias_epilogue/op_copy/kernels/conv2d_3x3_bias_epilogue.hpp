// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Bias epilogue on the K-blocked walk (the descriptor's BIAS_EPI): the K walk of one M-block without the separate
// fp32-partials + bias pass after it.
//
// K-blocks 0 .. K-2 pack the fp32 partials into the caller-owned [bm, bn] region (packer L1 accumulation from K-block 1
// on, block 0 overwrites), tile-row major, no FIFO traffic. Then, by mode:
//
//   Add         After K-block K-2 the packer pushes the region once. The last K-block, per output subblock: DEST =
//               partials + bias row (one FPU row-broadcast add per tile, partials -> SrcA, bias -> SrcB), the last
//               K-block's matmul accumulates on top, the final values pack straight to the output CB (per row group
//               reserve / push). No fp32 pack of the last K-block, no second pass; the output packs interleave with the
//               last K-block's matmuls. Not bit-identical: the bias joins the partial sum of K-blocks 0 .. K-2
//               (truncated to the SrcA format on the reload), the last K-block accumulates in fp32 DEST on top.
//   Copy        As Add without a bias (the no-bias walk): DEST = partials (plain copy). Same arithmetic as
//               compute_kernel_lib::matmul_block's Out target: bit-identical to it.
//   Seed        As Copy, and K-block 0 starts each subblock's DEST at the bias row (a row-broadcast copy of the bias
//               tile) instead of zero. With the partials CB unpacked straight to DEST (the host's UnpackToDestFp32)
//               the reload is exact fp32: more precise than the bias pass, never rounded below fp32 before the output.
//   Interleave  The last K-block packs the partials as every other K-block does (L1 accumulation), pushing them one row
//               group at a time, and the bias pass runs inside it: after every `groups` row groups of matmul, the
//               previous `groups` row groups get the bias add (partials + bias row -> output CB). Same packs and adds
//               as the op's bias pass, in a different order: bit-identical; the output stores overlap the last
//               K-block's matmuls instead of trailing the walk.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block reloads the partials only under its own FIFO bookkeeping
// (caller_owns_pack_target is static_assert'ed to Interm), so its last K-block cannot reload from a caller-owned
// region; its reload is a plain copy (no hook adds the bias there); and compute_kernel_lib::add_bias_bcast_rows runs
// over whole M-blocks only (no row-group interleave with the matmul). Bypassed: compute_kernel_lib::matmul_block on
// this walk and compute_kernel_lib::add_bias_bcast_rows after it. Same LLK calls as those helpers (matmul_block_init,
// ckernel::matmul_block, pack_tile<true>, copy_block, add_bcast_rows_init / add_tiles_bcast_rows, unary_bcast).

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::bias_epilogue {

enum class Mode : uint32_t { Copy = 0, Add = 1, Seed = 2, Interleave = 3 };

template <
    Mode mode,
    uint32_t groups,    // Interleave: row groups per bias batch
    bool in1_resident,  // in1 NoWaitNoPop (weights resident: the caller waits, In1BaseOffsetFn picks the round)
    bool relu,
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t partials_cb,
    uint32_t out_cb,
    uint32_t bias_cb,
    uint32_t bias_tiles,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t bk,
    uint32_t num_k_blocks,
    typename PreKBlockFn,
    typename In0SourceFn,
    typename In1BaseOffsetFn>
ALWI void walk(const PreKBlockFn& pre_k_block, const In0SourceFn& in0_source, const In1BaseOffsetFn& in1_offset) {
    static_assert(num_k_blocks >= 2, "one K-block has no partials to reload");
    static_assert(mode != Mode::Interleave || groups >= 1);
    constexpr bool reload = mode != Mode::Interleave;
    constexpr bool seed = mode == Mode::Seed;
    constexpr uint32_t bn = in1_num_subblocks * sb_w;
    constexpr uint32_t row_group_tiles = sb_h * bn;
    constexpr uint32_t block_tiles = in0_num_subblocks * row_group_tiles;
    constexpr uint32_t in0_block_tiles = in0_num_subblocks * sb_h * bk;
    constexpr uint32_t in1_block_tiles = bk * bn;

    // Interleave: `n` row groups of the (pushed) partials -> + bias -> output CB, the matmul state restored after.
    auto bias_rows = [&](uint32_t n, uint32_t a_cb) {
        PACK((pack_reconfig_data_format(out_cb)));
        PACK((llk_pack_reconfig_l1_acc(0)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(ReluConfig::zero())));
        }
        reconfig_data_format_srca(in1_cb, partials_cb);
        reconfig_data_format_srcb(a_cb, bias_cb);
        add_bcast_rows_init(partials_cb, bias_cb);
        for (uint32_t g = 0; g < n; ++g) {
            cb_wait_front(partials_cb, row_group_tiles);
            cb_reserve_back(out_cb, row_group_tiles);
            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {
                const uint32_t col = i1 * sb_w;
                tile_regs_acquire();
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t c = 0; c < sb_w; ++c) {
                        add_tiles_bcast_rows(partials_cb, bias_cb, r * bn + col + c, col + c, r * sb_w + c);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                compute_kernel_lib::pack_subblock_row_strided(0, out_cb, col, bn, sb_h, sb_w);
                tile_regs_release();
            }
            cb_pop_front(partials_cb, row_group_tiles);
            cb_push_back(out_cb, row_group_tiles);
        }
        reconfig_data_format_srca(partials_cb, in1_cb);
        reconfig_data_format_srcb(bias_cb, a_cb);
        ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
        PACK((pack_reconfig_data_format(partials_cb)));
        PACK((llk_pack_reconfig_l1_acc(1)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
        }
    };

    // The partials region: one reserve per M-block (the previous M-block's last K-block popped it).
    cb_reserve_back(partials_cb, block_tiles);
    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        const bool last = block == num_k_blocks - 1;
        // The last K-block packs the output (reload modes) or keeps accumulating the partials (Interleave).
        const bool out_last = reload && last;
        const bool ilv_last = !reload && last;
        pre_k_block(block, num_k_blocks, last);
        const uint32_t a_cb = in0_source(block, in0_cb);
        // Matmul state after the hook (the tilize reconfigures unpack / math / pack).
        reconfig_data_format(in1_cb, a_cb);
        if (out_last) {
            PACK((pack_reconfig_data_format(out_cb)));
            // The output must not re-accumulate.
            PACK((llk_pack_reconfig_l1_acc(0)));
            if constexpr (relu) {
                PACK((llk_pack_relu_config(ReluConfig::zero())));
            }
        } else {
            PACK((pack_reconfig_data_format(partials_cb)));
            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
        }
        ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
        const uint32_t in1_base = in1_offset(block);
        cb_wait_front(a_cb, in0_block_tiles);
        if constexpr (!in1_resident) {
            cb_wait_front(in1_cb, in1_block_tiles);
        }
        // First use of the bias (it stays fronted until the kernel end; later waits return at once).
        if constexpr (seed) {
            if (block == 0) {
                cb_wait_front(bias_cb, bias_tiles);
            }
        }
        if (out_last) {
            // The packer pushed the region after K-block K-2's packs.
            cb_wait_front(partials_cb, block_tiles);
        }
        if constexpr (mode == Mode::Add || mode == Mode::Interleave) {
            if (last) {
                cb_wait_front(bias_cb, bias_tiles);
            }
        }
        uint32_t in0_sub = 0;
        uint32_t biased = 0;  // Interleave: row groups through the bias pass so far
        for (uint32_t i0 = 0; i0 < in0_num_subblocks; ++i0) {
            if (out_last) {
                cb_reserve_back(out_cb, row_group_tiles);
            }
            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {
                const uint32_t col = i1 * sb_w;
                tile_regs_acquire();
                if constexpr (seed) {
                    if (block == 0) {
                        // DEST = the bias row of each output column: K-block 0 accumulates on it.
                        reconfig_data_format_srca(in1_cb, bias_cb);
                        reconfig_data_format_srcb(a_cb, bias_cb);
                        unary_bcast_init<BroadcastType::ROW>(bias_cb);
                        for (uint32_t r = 0; r < sb_h; ++r) {
                            for (uint32_t c = 0; c < sb_w; ++c) {
                                unary_bcast<BroadcastType::ROW>(bias_cb, col + c, r * sb_w + c);
                            }
                        }
                        unary_bcast_uninit<BroadcastType::ROW>(bias_cb);
                        reconfig_data_format_srca(bias_cb, in1_cb);
                        reconfig_data_format_srcb(bias_cb, a_cb);
                        ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
                    }
                }
                if (out_last) {
                    // DEST = partials (+ bias row): the matmul below accumulates the last K-block on top.
                    reconfig_data_format_srca(in1_cb, partials_cb);
                    if constexpr (mode == Mode::Add) {
                        reconfig_data_format_srcb(a_cb, bias_cb);
                        add_bcast_rows_init(partials_cb, bias_cb);
                        for (uint32_t r = 0; r < sb_h; ++r) {
                            const uint32_t src = i0 * row_group_tiles + r * bn + col;
                            for (uint32_t c = 0; c < sb_w; ++c) {
                                add_tiles_bcast_rows(partials_cb, bias_cb, src + c, col + c, r * sb_w + c);
                            }
                        }
                        reconfig_data_format_srcb(bias_cb, a_cb);
                    } else {
                        copy_init(partials_cb);
                        compute_kernel_lib::copy_subblock_row_strided(
                            partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);
                    }
                    reconfig_data_format_srca(partials_cb, in1_cb);
                    ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
                }
                uint32_t in0_index = in0_sub;
                uint32_t in1_index = in1_base + col;
                for (uint32_t k = 0; k < bk; ++k) {
                    ckernel::matmul_block(a_cb, in1_cb, in0_index, in1_index, 0, false, sb_w, sb_h, bk);
                    ++in0_index;
                    in1_index += bn;
                }
                tile_regs_commit();
                tile_regs_wait();
                if (out_last) {
                    compute_kernel_lib::pack_subblock_row_strided(0, out_cb, col, bn, sb_h, sb_w);
                } else if (ilv_last) {
                    // Pushed one row group at a time: the write pointer sits at this row group.
                    compute_kernel_lib::pack_subblock_row_strided(0, partials_cb, col, bn, sb_h, sb_w);
                } else {
                    compute_kernel_lib::pack_subblock_row_strided(
                        0, partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);
                }
                tile_regs_release();
            }
            if (out_last) {
                cb_push_back(out_cb, row_group_tiles);
            }
            in0_sub += sb_h * bk;
            if constexpr (mode == Mode::Interleave) {
                if (ilv_last) {
                    cb_push_back(partials_cb, row_group_tiles);
                    // One batch behind the matmul: the batch's packs are done by the time the unpacker reads them.
                    if (i0 + 1 - biased >= 2 * groups) {
                        bias_rows(groups, a_cb);
                        biased += groups;
                    }
                }
            }
        }
        if constexpr (mode == Mode::Interleave) {
            if (ilv_last) {
                bias_rows(in0_num_subblocks - biased, a_cb);
            }
        }
        if constexpr (reload) {
            if (block + 2 == num_k_blocks) {
                cb_push_back(partials_cb, block_tiles);
            }
            if (out_last) {
                cb_pop_front(partials_cb, block_tiles);
            }
        }
        cb_pop_front(a_cb, in0_block_tiles);
        if constexpr (!in1_resident) {
            cb_pop_front(in1_cb, in1_block_tiles);
        }
    }
}

}  // namespace conv2d_3x3::bias_epilogue
