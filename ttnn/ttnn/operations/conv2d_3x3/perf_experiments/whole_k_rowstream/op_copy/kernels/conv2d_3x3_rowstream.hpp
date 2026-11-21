// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The row-streamed whole-K walk (the descriptor's ROWSTREAM: the ROW_MAJOR ring slab with resident weights).
//
// The core's tile-rows run in batches of `rows` (each M-block's last batch may be shorter): compute tilizes the batch's
// units in the reader's order (tile-row major, every K-block of a tile-row, so in0 is [rows, K] tile-row major), then
// accumulates each subblock over all of K in DEST against the resident weights (round k at tile offset k * bk * bn, so
// in1 is [K, bn] row major), packs it once and pushes each row group to the output CB as it completes. There are no
// fp32 partials and no per-K-block packs, so the output leaves evenly across the walk instead of in the last K-block.
// The next batch's tilize runs between a batch's matmul and its bias pass, so that pass finds every pack done.
//
// Bias (bias_mode): 1, the batch's fp32 sums go to cb_partials and the row-broadcast add (the K-blocked walk's bias
// pass) packs the output; 2, the bias is one more K-step at the start of every subblock: a tile whose column 0 is all
// ones (ones_cb, written once by the reader) times the bias tile (row 0 the bias), exact at HiFi4, so DEST starts at
// the bias and the matmul packs the output. 2 needs a subblock that reuses in0 (sb_w >= sb_h: the ones tile is read
// once). A batch shorter than `rows` still pushes and pops `rows` tile-rows of in0 (the extra ones unwritten and
// unread), so every batch's in0 starts at the CB base and the matmul's tile indices never wrap.
//
// Raw LLK (capability), as the K-blocked walk: compute_kernel_lib::matmul_block accumulates across K-blocks through
// packer L1 accumulation into a partials CB and has no in0 source per K-step (the bias K-step reads another CB);
// compute_kernel_lib::add_bias_bcast_rows owns its whole-block waits. Same LLK calls as those helpers
// (matmul_block_init, ckernel::matmul_block, pack_tile, add_bcast_rows_init, add_tiles_bcast_rows).

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::rowstream {

template <
    uint32_t bias_mode,  // 0: no bias, 1: add pass per batch, 2: bias K-step
    bool relu,
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t partials_cb,
    uint32_t out_cb,
    uint32_t bias_cb,
    uint32_t ones_cb,
    uint32_t in1_num_subblocks,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t K,
    uint32_t bm,
    uint32_t rows,
    typename TilizeFn,
    typename ReadyFn>
ALWI void walk(uint32_t num_m_blocks, const TilizeFn& tilize, const ReadyFn& ready) {
    static_assert(rows % sb_h == 0 && bm % sb_h == 0 && rows <= bm);
    static_assert(bias_mode != 2 || sb_w >= sb_h, "the bias K-step reads one ones tile per subblock");
    constexpr uint32_t bn = in1_num_subblocks * sb_w;
    constexpr uint32_t rg_tiles = sb_h * bn;
    constexpr uint32_t batches = (bm + rows - 1) / rows;
    constexpr uint32_t in0_tiles = rows * K;
    constexpr uint32_t mm_target = bias_mode == 1 ? partials_cb : out_cb;
    const uint32_t total = num_m_blocks * batches;
    auto rows_of = [](uint32_t b) -> uint32_t {
        const uint32_t i = b % batches;
        return (i + 1) * rows <= bm ? rows : bm - i * rows;
    };

    auto matmul = [&](uint32_t n) {
        reconfig_data_format(in1_cb, in0_cb);
        PACK((pack_reconfig_data_format(mm_target)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(bias_mode == 1 ? ReluConfig::none() : ReluConfig::zero())));
        }
        ckernel::matmul_block_init(in0_cb, in1_cb, false, sb_w, sb_h, K);
        cb_wait_front(in0_cb, in0_tiles);
        uint32_t in0_row = 0;
        for (uint32_t i0 = 0; i0 < n / sb_h; ++i0) {
            cb_reserve_back(mm_target, rg_tiles);
            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {
                const uint32_t col = i1 * sb_w;
                tile_regs_acquire();
                if constexpr (bias_mode == 2) {
                    ckernel::matmul_block(ones_cb, bias_cb, 0, col, 0, false, sb_w, sb_h, K);
                }
                uint32_t in0_index = in0_row;
                uint32_t in1_index = col;
                for (uint32_t k = 0; k < K; ++k) {
                    ckernel::matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, sb_w, sb_h, K);
                    ++in0_index;
                    in1_index += bn;
                }
                tile_regs_commit();
                tile_regs_wait();
                compute_kernel_lib::pack_subblock_row_strided(0, mm_target, col, bn, sb_h, sb_w);
                tile_regs_release();
            }
            cb_push_back(mm_target, rg_tiles);
            in0_row += sb_h * K;
        }
        cb_pop_front(in0_cb, in0_tiles);
    };

    // n tile-rows of fp32 sums + the bias row -> the output CB (ReLU armed for these packs only).
    auto bias_pass = [&](uint32_t n) {
        MaybeDeviceZoneScope("compute_bias_add");
        PACK((pack_reconfig_data_format(out_cb)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(ReluConfig::zero())));
        }
        reconfig_data_format(partials_cb, bias_cb);
        add_bcast_rows_init(partials_cb, bias_cb);
        for (uint32_t g = 0; g < n / sb_h; ++g) {
            cb_wait_front(partials_cb, rg_tiles);
            cb_reserve_back(out_cb, rg_tiles);
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
            cb_pop_front(partials_cb, rg_tiles);
            cb_push_back(out_cb, rg_tiles);
        }
        if constexpr (relu) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
        }
    };

    if (total == 0) {
        return;
    }
    tilize(rows_of(0));
    ready();
    for (uint32_t b = 0; b < total; ++b) {
        const uint32_t n = rows_of(b);
        matmul(n);
        if (b + 1 < total) {
            tilize(rows_of(b + 1));
        }
        if constexpr (bias_mode == 1) {
            bias_pass(n);
        }
    }
    if constexpr (relu) {
        PACK((llk_pack_relu_config(ReluConfig::none())));
    }
}

}  // namespace conv2d_3x3::rowstream
