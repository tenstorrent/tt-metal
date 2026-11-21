// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The K-blocked walk of one M-block (conv2d_3x3_compute.cpp: !k_split, !slab_direct, one in0 group), with the bias pass
// interleaved into the last K-block.
//
// Every K-block but the last packs the fp32 partials into the caller-owned [bm, bn] region (one reserve per M-block),
// tile-row major at absolute offsets, packer L1 accumulation from K-block 1 on (K-block 0 overwrites); nothing is
// pushed, so the unpacker never waits on those packs.
//
// With a bias, the last K-block packs onto the partials the same way, but pushes them one row group at a time, and the
// bias pass runs inside it: after every `groups` row groups of matmul, the previous `groups` row groups go through the
// row-broadcast bias add (fp32 partials -> SrcA, bias row -> SrcB, ReLU armed for that pack only) into the output CB.
// Same packs and adds as compute_kernel_lib::add_bias_bcast_rows after the whole walk, in a different order:
// bit-identical. The gain is the overlap: the bias pass used to start after the walk's last pack, so the output CB
// (OUT_DEPTH_ROW_GROUPS row groups) drained to DRAM only then, with the unpacker / FPU idle behind the writer; now the
// writer drains `groups` row groups while the next `groups` row groups multiply. groups = the output CB depth in row
// groups. Measured (Blackhole p150b, FLUX.2 bh2x4, device kernel ns, the walk's input fill stubbed / full op):
// P1 1026x2048 32->128 1808699 -> 1777377 / 2607956 -> 2586661, P2 514x1024 64->256 1422163 -> 1390070 /
// 1619113 -> 1599208 (groups 1: P1 1770401, P2 1386546; groups 4: 1779452, 1408737; groups 8: 1807514, 1417032).
//
// Without a bias, the last K-block reloads the partials into DEST (copy; the push after K-block K-2 is the unpacker's
// only wait), multiplies on top and packs the output: the arithmetic of compute_kernel_lib::matmul_block's Out target,
// bit-identical, without its per-K-block partials drain (the unpacker waited for every non-last K-block's last pack).
//
// One K-block (face-direct whole K): no partials reload; with a bias the single K-block is the interleaved last one.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block's caller_owns_pack_target is static_assert'ed to the Interm
// target and its last K-block offers no per-row-group hook, so it can neither reload from a caller-owned region (Out)
// nor push the last K-block's partials row group by row group for an interleaved consumer; compute_kernel_lib::
// add_bias_bcast_rows runs over whole blocks of row groups waited for at once. Bypassed: both helpers on this walk.
// Same LLK calls as those helpers (matmul_block_init, ckernel::matmul_block, pack_tile<true>, copy_block,
// add_bcast_rows_init, add_tiles_bcast_rows).

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::kblock_walk {

template <
    bool has_bias,
    uint32_t groups,    // with a bias: row groups per interleaved bias batch
    bool in1_resident,  // in1 NoWaitNoPop (weights resident: the PreKBlockFn waits, In1BaseOffsetFn picks the round)
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
    static_assert(!has_bias || groups >= 1);
    constexpr uint32_t bn = in1_num_subblocks * sb_w;
    constexpr uint32_t row_group_tiles = sb_h * bn;
    constexpr uint32_t block_tiles = in0_num_subblocks * row_group_tiles;
    constexpr uint32_t in0_block_tiles = in0_num_subblocks * sb_h * bk;
    constexpr uint32_t in1_block_tiles = bk * bn;
    // No bias: the last K-block reloads the partials of the others (none with one K-block).
    constexpr bool reload = !has_bias && num_k_blocks >= 2;
    constexpr bool uses_partials = has_bias || reload;

    // `n` pushed row groups of the partials -> + bias row -> the output CB; then the matmul state of K-block `block`.
    auto bias_rows = [&](uint32_t n, uint32_t a_cb, uint32_t block) {
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
#ifndef CONV2D_3X3_ABLATE_BIAS
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
#endif
            cb_pop_front(partials_cb, row_group_tiles);
            cb_push_back(out_cb, row_group_tiles);
        }
        reconfig_data_format_srca(partials_cb, in1_cb);
        reconfig_data_format_srcb(bias_cb, a_cb);
        ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
        PACK((pack_reconfig_data_format(partials_cb)));
        PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
        }
    };

    auto run_block = [&](uint32_t block) {
        const bool last = block == num_k_blocks - 1;
        // No bias: the last K-block packs the output. With a bias every K-block packs the partials.
        const bool out_last = !has_bias && last;
        const bool bias_last = has_bias && last;
        pre_k_block(block, num_k_blocks, last);
        const uint32_t a_cb = in0_source(block, in0_cb);
#ifdef TU_ZI
        {
            DeviceZoneScopedN("tu_mminit");
#endif
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
#ifdef TU_ZI
        }
        {
            DeviceZoneScopedN("tu_mmwait");
            cb_wait_front(a_cb, in0_block_tiles);
        }
#else
        cb_wait_front(a_cb, in0_block_tiles);
#endif
        const uint32_t in1_base = in1_offset(block);
        if constexpr (!in1_resident) {
            cb_wait_front(in1_cb, in1_block_tiles);
        }
        if constexpr (reload) {
            if (last) {
                // The packer pushed the region after K-block K-2's packs.
                cb_wait_front(partials_cb, block_tiles);
            }
        }
        if constexpr (has_bias) {
            if (last) {
                // First use of the bias (the writer reads it at kernel start; it stays fronted until the kernel end).
                MaybeDeviceZoneScope("compute_bias_wait");
                cb_wait_front(bias_cb, bias_tiles);
            }
        }
        uint32_t in0_sub = 0;
        uint32_t biased = 0;  // row groups through the bias pass so far (last K-block, with a bias)
        for (uint32_t i0 = 0; i0 < in0_num_subblocks; ++i0) {
            if (out_last) {
                cb_reserve_back(out_cb, row_group_tiles);
            }
#ifndef CONV2D_3X3_ABLATE_MATMUL
            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {
                const uint32_t col = i1 * sb_w;
                tile_regs_acquire();
                if constexpr (reload) {
                    if (last) {
                        // DEST = the partials: the matmul below accumulates the last K-block on top.
                        reconfig_data_format_srca(in1_cb, partials_cb);
                        copy_init(partials_cb);
                        compute_kernel_lib::copy_subblock_row_strided(
                            partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);
                        reconfig_data_format_srca(partials_cb, in1_cb);
                        ckernel::matmul_block_init(a_cb, in1_cb, false, sb_w, sb_h, bk);
                    }
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
                } else if (bias_last) {
                    // Pushed one row group at a time: the write pointer sits at this row group.
                    compute_kernel_lib::pack_subblock_row_strided(0, partials_cb, col, bn, sb_h, sb_w);
                } else {
                    compute_kernel_lib::pack_subblock_row_strided(
                        0, partials_cb, i0 * row_group_tiles + col, bn, sb_h, sb_w);
                }
                tile_regs_release();
            }
#endif
            if (out_last) {
                cb_push_back(out_cb, row_group_tiles);
            }
            in0_sub += sb_h * bk;
            if constexpr (has_bias) {
                if (last) {
                    cb_push_back(partials_cb, row_group_tiles);
                    // One batch behind the matmul: the batch's packs are done by the time the unpacker reads them.
                    if (i0 + 1 - biased >= 2 * groups) {
                        bias_rows(groups, a_cb, block);
                        biased += groups;
                    }
                }
            }
        }
        if constexpr (has_bias) {
            if (last) {
                bias_rows(in0_num_subblocks - biased, a_cb, block);
            }
        }
        if constexpr (reload) {
            if (block + 2 == num_k_blocks) {
                cb_push_back(partials_cb, block_tiles);
            }
            if (last) {
                cb_pop_front(partials_cb, block_tiles);
            }
        }
        cb_pop_front(a_cb, in0_block_tiles);
        if constexpr (!in1_resident) {
            cb_pop_front(in1_cb, in1_block_tiles);
        }
    };

    if constexpr (uses_partials) {
        // The partials region, once per M-block (the previous M-block's last K-block popped it).
        cb_reserve_back(partials_cb, block_tiles);
    }
    for (uint32_t block = 0; block + 1 < num_k_blocks; ++block) {
        run_block(block);
    }
    {
        // The last K-block: the output packs (and, with a bias, the interleaved bias pass).
        MaybeDeviceZoneScope("compute_last_kblock");
        run_block(num_k_blocks - 1);
    }
}

}  // namespace conv2d_3x3::kblock_walk
