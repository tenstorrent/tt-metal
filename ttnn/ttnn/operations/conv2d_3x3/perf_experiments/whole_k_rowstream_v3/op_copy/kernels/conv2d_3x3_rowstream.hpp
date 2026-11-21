// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The row-streamed whole-K walk (the descriptor's ROWSTREAM: the ROW_MAJOR ring slab with resident weights).
//
// The core's valid output tile-rows run flat in batches of `rows` (no M-blocks: a core whose tile-rows are not a
// multiple of a block extent multiplies no padding rows). Compute tilizes the batch's units, then accumulates each
// subblock over all of K in DEST against the resident weights (in1 is [K, bn] row major, round k at tile offset
// k * bk * bn), packs it once and pushes each row group as it completes. There are no fp32 partials and no per-K-block
// packs, so the output leaves evenly across the walk instead of in an M-block's last K-block. The next batch's tilize
// runs between a batch's matmul and its bias pass, so that pass finds every pack done.
//
// in0 (BlockIn0): the batch's [rows, K] block at the CB read pointer (the units tilized tile-row major, every K-block
// of a tile-row). The last batch may be shorter: it still pushes and pops `rows` tile-rows of in0 (the extra ones
// unwritten and unread), so every batch's in0 starts at the CB base and the matmul's tile indices never wrap.
//
// Bias: the batch's fp32 sums go to cb_partials and the row-broadcast add (the K-blocked walk's bias pass) packs the
// output, one row group at a time. Without a bias the matmul packs the output.
//
// Raw LLK (capability), as the K-blocked walk: compute_kernel_lib::matmul_block accumulates across K-blocks through
// packer L1 accumulation into a partials CB (here all of K stays in DEST, one pack per subblock);
// compute_kernel_lib::add_bias_bcast_rows owns its whole-block waits (here per row group, so each group's output
// leaves as it is added). Same LLK calls as those helpers (matmul_block_init, ckernel::matmul_block, pack_tile,
// add_bcast_rows_init, add_tiles_bcast_rows).

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::rowstream {

// The batch's [rows, K] in0 block at the CB read pointer (tile-row major).
template <uint32_t in0_cb, uint32_t K, uint32_t rows>
struct BlockIn0 {
    static constexpr uint32_t cb = in0_cb;
    ALWI void wait() const { cb_wait_front(in0_cb, rows * K); }
    ALWI void pop() const { cb_pop_front(in0_cb, rows * K); }
    // Tile index of the first K-step of kernel row kh, batch tile-row r.
    ALWI uint32_t base(uint32_t r, uint32_t kh) const { return r * K + kh * (K / 3); }
};

template <
    bool has_bias,
    bool relu,
    uint32_t in1_cb,
    uint32_t partials_cb,
    uint32_t out_cb,
    uint32_t bias_cb,
    uint32_t in1_num_subblocks,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t K,
    uint32_t rows,
    typename In0,
    typename TilizeFn,
    typename ReadyFn>
ALWI void walk(uint32_t total_rows, const In0& in0, const TilizeFn& tilize, const ReadyFn& ready) {
    static_assert(sb_h == 1, "a row group is one tile-row: batches and the last one end on any tile-row");
    static_assert(K % 3 == 0);
    constexpr uint32_t in0_cb = In0::cb;
    constexpr uint32_t kk = K / 3;  // K-steps per kernel row
    constexpr uint32_t bn = in1_num_subblocks * sb_w;
    constexpr uint32_t rg_tiles = sb_h * bn;
    constexpr uint32_t mm_target = has_bias ? partials_cb : out_cb;
    const uint32_t total = (total_rows + rows - 1) / rows;
    auto rows_of = [&](uint32_t b) -> uint32_t { return (b + 1) * rows <= total_rows ? rows : total_rows - b * rows; };

    auto matmul = [&](uint32_t n) {
        reconfig_data_format(in1_cb, in0_cb);
        PACK((pack_reconfig_data_format(mm_target)));
        if constexpr (relu) {
            PACK((llk_pack_relu_config(has_bias ? ReluConfig::none() : ReluConfig::zero())));
        }
        ckernel::matmul_block_init(in0_cb, in1_cb, false, sb_w, sb_h, K);
        in0.wait();
        for (uint32_t i0 = 0; i0 < n / sb_h; ++i0) {
            cb_reserve_back(mm_target, rg_tiles);
            for (uint32_t i1 = 0; i1 < in1_num_subblocks; ++i1) {
                const uint32_t col = i1 * sb_w;
                tile_regs_acquire();
                uint32_t in1_index = col;
                for (uint32_t kh = 0; kh < 3; ++kh) {
                    uint32_t in0_index = in0.base(i0 * sb_h, kh);
                    for (uint32_t k = 0; k < kk; ++k) {
                        ckernel::matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, sb_w, sb_h, K);
                        ++in0_index;
                        in1_index += bn;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                compute_kernel_lib::pack_subblock_row_strided(0, mm_target, col, bn, sb_h, sb_w);
                tile_regs_release();
            }
            cb_push_back(mm_target, rg_tiles);
        }
        in0.pop();
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
        if constexpr (has_bias) {
            bias_pass(n);
        }
    }
    if constexpr (relu) {
        PACK((llk_pack_relu_config(ReluConfig::none())));
    }
}

}  // namespace conv2d_3x3::rowstream
