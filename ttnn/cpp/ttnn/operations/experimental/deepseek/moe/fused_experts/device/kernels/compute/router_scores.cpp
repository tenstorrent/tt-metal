// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

// Router compute kernel (runs on the router cores, the row below the expert grid).
//
// For each of this core's n_tiles gate tile columns n -- global column first_tile + n * col_stride,
// resident in cb_rw as tiles [n * k_tiles, +k_tiles) -- score = sqrt(softplus(x @ W[:, col])) and
// ranking = score + bias[col]. The matmul is matmul_decode's custom_mm (LoFi, 16-bit DEST); scores sit
// in dst n, rankings in dst n_tiles + n, and are packed to cb_rout as bf16 [scores..., rankings...].
// Every tile is 1x32, so the SFPU ops only touch the first rows of faces 0 and 1 (VectorMode::R, two
// iterations per face), as deepseek_v3_b1's matmul does for its fused activation.
//
// Compile-time args:
//   0: cb_rin   1: k_tiles   2: cb_rw   3: cb_rbias   4: bias_tiles   5: cb_rout   6: col_stride
//   7: piece_tiles (the K tiles per weight piece the reader pushes; k_tiles for an L1 gate)
//
// Runtime args:
//   0: first_tile (this core's first tile column)   1: n_tiles

namespace {
constexpr uint32_t kOneBits = 0x3f800000u;        // softplus beta and 1/beta: 1.0f
constexpr uint32_t kThresholdBits = 0x41a00000u;  // 20.0f, ttnn.softplus's default threshold
constexpr int kTinyTileIterations = 2;            // a 1x32 tile's rows of a face
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rin_id = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rw_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rbias_id = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rout_id = get_compile_time_arg_val(5);
    constexpr uint32_t col_stride = get_compile_time_arg_val(6);
    constexpr uint32_t piece_tiles = get_compile_time_arg_val(7);

    const uint32_t first_tile = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    CircularBuffer in_cb(cb_rin_id);
    CircularBuffer w_cb(cb_rw_id);
    CircularBuffer bias_cb(cb_rbias_id);
    CircularBuffer out_cb(cb_rout_id);

    constexpr bool transpose = false;
    // Standard 64-row DEST tile stride, so dst n is the same tile for the matmul, SFPU and packer.
    constexpr auto layout = CustomMmDestLayout::DefaultSplitAcc;

    // SrcOrder::Reverse: SrcA holds the weight (in1), SrcB the activation (in0).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_rin_id, cb_rw_id, cb_rout_id);

    out_cb.reserve_back(2u * n_tiles);
    tile_regs_acquire();

    // The bias goes into dst n_tiles + n first, so custom_mm is the last FPU op ahead of the SFPU tail.
    // reconfig_full_operand_srca, not reconfig_data_format_srca: SrcA's face geometry changes between the
    // 32x32 weight and the 1x32 bias, and a format-only reconfig would unpack one as the other.
    bias_cb.wait_front(bias_tiles);
    copy_init(cb_rbias_id);
    reconfig_full_operand_srca(cb_rbias_id);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        copy_tile(cb_rbias_id, first_tile + n * col_stride, n_tiles + n);
    }
    reconfig_full_operand_srca(cb_rw_id);

    custom_mm_block_init_short<transpose, layout>(cb_rin_id, cb_rw_id, cb_rout_id, 1);
    in_cb.wait_front(k_tiles);
    {
        DeviceZoneScopedN("FE_R_MM");
        // A DRAM gate arrives piece by piece (an L1 one all at once); only a column's last piece finalizes.
        constexpr uint32_t pieces_per_col = k_tiles / piece_tiles;
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t p = 0; p < pieces_per_col; ++p) {
                const uint32_t k0 = p * piece_tiles;
                w_cb.wait_front(n * k_tiles + k0 + piece_tiles);
                if (p + 1u == pieces_per_col) {
                    custom_mm_block<layout, true>(cb_rin_id, cb_rw_id, k0, n * k_tiles + k0, n, piece_tiles, 1);
                } else {
                    custom_mm_block<layout, false>(cb_rin_id, cb_rw_id, k0, n * k_tiles + k0, n, piece_tiles, 1);
                }
            }
        }
    }
    custom_mm_block_uninit<layout>(cb_rout_id);

    softplus_tile_init();
    for (uint32_t n = 0; n < n_tiles; ++n) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_softplus,
            (APPROX, DST_ACCUM_MODE, kTinyTileIterations),
            n,
            VectorMode::R,
            kOneBits,
            kOneBits,
            kThresholdBits));
    }
    sqrt_tile_init();
    for (uint32_t n = 0; n < n_tiles; ++n) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sqrt,
            (APPROX, kTinyTileIterations, DST_ACCUM_MODE, false),
            n,
            VectorMode::R));
    }

    add_binary_tile_init();
    for (uint32_t n = 0; n < n_tiles; ++n) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::ADD, kTinyTileIterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            n,
            n_tiles + n,
            n_tiles + n,
            VectorMode::R)));
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t t = 0; t < 2u * n_tiles; ++t) {
        pack_tile<true>(t, cb_rout_id, t);
    }
    tile_regs_release();
    out_cb.push_back(2u * n_tiles);

    bias_cb.pop_front(bias_tiles);
    w_cb.pop_front(n_tiles * k_tiles);
    in_cb.pop_front(k_tiles);
}
