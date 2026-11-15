// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/experimental/rope_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// The RMSNorm and RoPE epilogue on 1x32 tiles, after custom_mm has left the packer in its own
// configuration (callers re-run compute_kernel_hw_startup first).

// The epilogue's multiplies run HiFi4 whatever the program's fidelity: custom_mm runs LoFi, whose
// truncated operands bias every square, and so every scale, low.

// scale_cb[0][0] = rsqrt(mean(x^2) + eps) over row tiles [tile_start, tile_start + tile_count).
ALWI void rms_scale(
    uint32_t row_cb_id,
    uint32_t stat_cb_id,
    uint32_t scale_cb_id,
    uint32_t tile_start,
    uint32_t tile_count,
    uint32_t inv_n_bits,
    uint32_t eps_bits) {
    CircularBuffer stat_cb(stat_cb_id);
    CircularBuffer scale_cb(scale_cb_id);

    reconfig_data_format(row_cb_id, row_cb_id);
    pack_reconfig_data_format(stat_cb_id);
    stat_cb.reserve_back(1);
    tile_regs_acquire();
    // The multiply accumulates into DST, and acquiring it does not guarantee a cleared slot.
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(row_cb_id, row_cb_id);
    mul_reduce_scalar_tile_fidelity<MathFidelity::HiFi4, MathFidelity::HiFi4, true>(
        row_cb_id, row_cb_id, stat_cb_id, tile_count, 1.0F, tile_start);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, stat_cb_id);
    mul_reduce_scalar_uninit();
    tile_regs_release();
    stat_cb.push_back(1);

    stat_cb.wait_front(1);
    scale_cb.reserve_back(1);
    reconfig_data_format(stat_cb_id, stat_cb_id);
    pack_reconfig_data_format(scale_cb_id);
    tile_regs_acquire();
    copy_init(stat_cb_id);
    copy_tile(stat_cb_id, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    add_rsqrt_tile_init();
    add_rsqrt_tile<false, VectorMode::RC_custom, 1>(0, eps_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scale_cb_id);
    tile_regs_release();
    scale_cb.push_back(1);
    stat_cb.pop_front(1);
}

// Appends src tiles [src_start, src_start + num_tiles) times scale_cb[0][0] (and, has_gamma, times
// gamma tiles [0, num_tiles)) to out_cb, which the caller has reserved. scale_cb must be at its front.
template <uint32_t num_tiles, bool has_gamma>
ALWI void scale_tiles(
    uint32_t src_cb_id, uint32_t src_start, uint32_t scale_cb_id, uint32_t gamma_cb_id, uint32_t out_cb_id) {
    reconfig_data_format(scale_cb_id, scale_cb_id);
    pack_reconfig_data_format(out_cb_id);
    pack_block_contiguous_init(out_cb_id);
    tile_regs_acquire();
    copy_init(scale_cb_id);
    copy_tile(scale_cb_id, 0, 0);
    reconfig_data_format_srca(scale_cb_id, src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4>(
        src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4, true>(
        src_cb_id, src_start, 0, 0);
    if constexpr (has_gamma) {
        reconfig_data_format(gamma_cb_id, gamma_cb_id);
        mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb_id);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(
                gamma_cb_id, t, t);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, out_cb_id, num_tiles);
    tile_regs_release();
}

// Appends the interleaved rotation of src tiles [0, rope_tiles) to out_cb, which the caller has
// reserved. cs holds rope_tiles cos tiles, then rope_tiles sin tiles, each angle duplicated across
// its pair. rows: the tiles' height; row r of x turns by row r of cos / sin (one SFPU pass covers four
// rows, so an 8-row tile takes a second pass over rows 4..7).
template <uint32_t rope_tiles, uint32_t rows = 1>
ALWI void rope_tiles_to(uint32_t src_cb_id, uint32_t cs_cb_id, uint32_t out_cb_id) {
    CircularBuffer src_cb(src_cb_id);
    CircularBuffer cs_cb(cs_cb_id);
    src_cb.wait_front(rope_tiles);
    cs_cb.wait_front(2 * rope_tiles);
    reconfig_data_format(src_cb_id, src_cb_id);
    pack_reconfig_data_format(out_cb_id);
    pack_init(out_cb_id);
    tile_regs_acquire();
    copy_init(src_cb_id);
    for (uint32_t t = 0; t < rope_tiles; ++t) {
        copy_tile(src_cb_id, t, t);
    }
    copy_init(cs_cb_id);
    for (uint32_t t = 0; t < 2 * rope_tiles; ++t) {
        copy_tile(cs_cb_id, t, rope_tiles + t);
    }
    rope_sfpu_fused_init();
    rope_sfpu_init();
    rope_sfpu_inplace<1, rope_tiles>();
    if constexpr (rows > 4) {
        constexpr uint32_t T = ROPE_SFPU_TILE_ROWS;
        rope_sfpu_inplace_rows<1, rope_tiles, 4, T, rope_tiles * T + 4, 2 * rope_tiles * T + 4, T>(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < rope_tiles; ++t) {
        pack_tile(t, out_cb_id);
    }
    tile_regs_release();
    src_cb.pop_front(rope_tiles);
    cs_cb.pop_front(2 * rope_tiles);
}

// ---- Several-row blocks: (rows x 32) tiles, one token per row ----
//
// A whole-tile scalar reduce would mix the rows, so the per-row RMSNorm statistic is
//   sq    = sum over the row's tiles of x * x            (elementwise, HiFi4, in one DST tile)
//   stat  = sq @ ones[32, 32]                             (custom_mm: each row's sum in all 32 columns)
//   scale = rsqrt(stat / n + eps)                         (each row's scale in all 32 columns)
// and normalizing is an elementwise multiply by scale.

// scale_cb[0] = per-row rsqrt(mean(x^2) + eps) over row tiles [tile_start, tile_start + tile_count), for (rows x 32)
// tiles; inv_n_bits is 1 / n as fp32 bits. ones_cb holds a 32x32 tile of ones at its front.
template <uint32_t rows, uint32_t inv_n_bits>
ALWI void rms_scale_rows(
    uint32_t row_cb_id,
    uint32_t sq_cb_id,
    uint32_t ones_cb_id,
    uint32_t stat_cb_id,
    uint32_t scale_cb_id,
    uint32_t tile_start,
    uint32_t tile_count,
    uint32_t eps_bits) {
    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;
    CircularBuffer sq_cb(sq_cb_id);
    CircularBuffer ones_cb(ones_cb_id);
    CircularBuffer stat_cb(stat_cb_id);
    CircularBuffer scale_cb(scale_cb_id);

    // sq: the squares, summed over the tiles elementwise. The multiply accumulates into DST.
    compute_kernel_hw_startup(row_cb_id, row_cb_id, sq_cb_id);
    sq_cb.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(row_cb_id, row_cb_id);
    for (uint32_t t = 0; t < tile_count; ++t) {
        UNPACK((llk_unpack_AB(row_cb_id, row_cb_id, tile_start + t, tile_start + t)));
        MATH((llk_math_eltwise_mul_reduce_scalar<DST_ACCUM_MODE, MathFidelity::HiFi4>(0, row_cb_id)));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_init(sq_cb_id);
    pack_tile(0, sq_cb_id);
    tile_regs_release();
    sq_cb.push_back(1);

    // stat: the row sums across the 32 columns, spread over them. The ones are exact in LoFi's SrcA.
    sq_cb.wait_front(1);
    ones_cb.wait_front(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(sq_cb_id, ones_cb_id, stat_cb_id);
    custom_mm_block_init_short<false, layout>(sq_cb_id, ones_cb_id, stat_cb_id, 1);
    pack_block_contiguous_init(stat_cb_id);
    stat_cb.reserve_back(1);
    tile_regs_acquire();
    custom_mm_block<layout, true>(sq_cb_id, ones_cb_id, 0, 0, 0, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, stat_cb_id, 1);
    tile_regs_release();
    custom_mm_block_uninit<layout>(stat_cb_id);
    stat_cb.push_back(1);
    sq_cb.pop_front(1);

    // scale. custom_mm_block_uninit leaves a 32x32 packer stride behind; reprogram for the row tiles.
    compute_kernel_hw_startup(stat_cb_id, stat_cb_id, scale_cb_id);
    stat_cb.wait_front(1);
    scale_cb.reserve_back(1);
    tile_regs_acquire();
    copy_init(stat_cb_id);
    copy_tile(stat_cb_id, 0, 0);
    // Only the block's rows of the two upper faces are live: an SFPU pass covers four rows of one column parity.
    constexpr int live_iterations = 2 * ((rows + 3) / 4);
    add_rsqrt_tile_init();
    add_rsqrt_tile<false, VectorMode::R, live_iterations, false, inv_n_bits>(0, eps_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_init(scale_cb_id);
    pack_tile(0, scale_cb_id);
    tile_regs_release();
    scale_cb.push_back(1);
    stat_cb.pop_front(1);
}

// Appends src tiles [src_start, src_start + num_tiles) times scale_cb[0] elementwise (each row by its
// own scale) and, has_gamma, times gamma tiles [0, num_tiles), to out_cb, which the caller has reserved.
template <uint32_t num_tiles, bool has_gamma>
ALWI void scale_rows_tiles(
    uint32_t src_cb_id, uint32_t src_start, uint32_t scale_cb_id, uint32_t gamma_cb_id, uint32_t out_cb_id) {
    reconfig_data_format(src_cb_id, src_cb_id);
    pack_reconfig_data_format(out_cb_id);
    pack_block_contiguous_init(out_cb_id);
    tile_regs_acquire();
    copy_init(src_cb_id);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        copy_tile(src_cb_id, src_start + t, t);
    }
    reconfig_data_format(scale_cb_id, scale_cb_id);
    mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(scale_cb_id);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(scale_cb_id, 0, t);
    }
    if constexpr (has_gamma) {
        reconfig_data_format(gamma_cb_id, gamma_cb_id);
        mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb_id);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(
                gamma_cb_id, t, t);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, out_cb_id, num_tiles);
    tile_regs_release();
}
