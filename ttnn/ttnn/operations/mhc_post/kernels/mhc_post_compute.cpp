// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute.
//
// mix_block, per output stream j, one eltwise_chain over the block's valid column tiles; per tile:
//   D1 <- F[c]                    D2 <- post_j (expanded)       D0 = D1 * D2           (SFPU fp32 mul)
//   for i in 0..n-1:  D1 <- X_i[c]   D2 <- comb[i][j] (expanded)   D0 = D0 + 1.0*D1*D2  (SFPU fp32 MAD)
//   pack D0 -> out slot j*B + c
// All fp32 CBs read via copy_tile are tagged UnpackToDestFp32 on the host (lossless); no FPU math.
// CB lifecycles are caller-managed (None, None) with TileAddressing::Offset bases: waited / reserved once
// per block (coefficients once per segment), popped / pushed with nominal counts.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "mhc_post_common.hpp"

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t n = get_compile_time_arg_val(0);
constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(3);
constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(4);
constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(5);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(6);

constexpr uint32_t FP32_ONE_BITS = 0x3F800000u;  // Addcmul scalar value = 1.0f (exact)

using SublayerCopy = CopyTile<
    input(
        cb_sublayer_tiles,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct),
    Dst::D1>;
using ResidualCopy = CopyTile<
    input(
        cb_residual_tiles,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset),
    Dst::D1>;
using CoefCopy = CopyTile<
    input(
        cb_coef_bcast,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Scalar,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset),
    Dst::D2>;
using ScaleSublayer = MulBinary<Dst::D1, Dst::D2, Dst::D0>;
using AccumulateStream = Addcmul<DataFormat::Float32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>;
using PackOutput = PackTile<
    output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Offset),
    Dst::D0>;

// Compile-time expansion of the n contraction terms (i = I .. n-1), then the terminal pack.
template <uint32_t I, class... Es>
ALWI void mix_stream(uint32_t j, uint32_t valid_col_tiles, Es... elts) {
    if constexpr (I == n) {
        eltwise_chain(IterationShape::tiles(valid_col_tiles), elts..., PackOutput{j * block_col_tiles});
    } else {
        mix_stream<I + 1>(
            j,
            valid_col_tiles,
            elts...,
            ResidualCopy{I * block_col_tiles},
            CoefCopy{n + I * n + j},
            AccumulateStream{FP32_ONE_BITS});
    }
}

// mix_block: all n output streams of one block.
ALWI void mix_block(uint32_t valid_col_tiles) {
    for (uint32_t j = 0; j < n; ++j) {
        mix_stream<0>(j, valid_col_tiles, SublayerCopy{}, CoefCopy{j}, ScaleSublayer{});
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_coef_tiles = n + n * n;
    constexpr uint32_t residual_block_tiles = n * block_col_tiles;
    constexpr uint32_t output_block_tiles = n * block_col_tiles;

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_residual_tiles, cb_coef_bcast, cb_output_tiles);

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        const uint32_t blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);

        cb_wait_front(cb_coef_bcast, num_coef_tiles);  // this row's expanded coefficient set
        for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
            const uint32_t valid = mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, block_idx);
            cb_wait_front(cb_sublayer_tiles, block_col_tiles);
            cb_wait_front(cb_residual_tiles, residual_block_tiles);
            cb_reserve_back(cb_output_tiles, output_block_tiles);

            mix_block(valid);

            cb_push_back(cb_output_tiles, output_block_tiles);
            cb_pop_front(cb_residual_tiles, residual_block_tiles);
            cb_pop_front(cb_sublayer_tiles, block_col_tiles);
        }
        cb_pop_front(cb_coef_bcast, num_coef_tiles);  // release_coefficients
    }
}
