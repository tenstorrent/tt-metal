// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute — plain-2d regime, mix_block via eltwise_chain (SFPU-only, fp32 DEST).
//
// Per (j, c) output tile of a block (IterationShape::grid(n, extent): row = j, col = c):
//   D1 <- bcast_col(post_j)          D2 <- F_c             D0 = D1 * D2          (MulBinary)
//   for i: D1 <- bcast_col(comb_ij)  D2 <- X_{i,c}         D0 = D0 + 1.0*D1*D2   (Addcmul, fused MAD)
//   pack D0 -> cb_out_tiles (per tile)
// The coefficient tiles carry their value in column 0 (the reader's expansion), so every
// consumer is a Col broadcast; post_j sits at tile j, comb[i][j] at n + i*n + j (comb is applied
// transposed). F / X / coefficients are caller-managed (None/None): F and X are waited for the
// whole nominal block and popped after the chain, cb_coef_cols is waited once per row and popped
// at the row end (release_coefficients).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(0);
constexpr uint32_t cb_f_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_c = get_compile_time_arg_val(5);
constexpr uint32_t n_coef_cols = get_compile_time_arg_val(6);

static_assert(n == 4, "mhc_post: the mix chain is written for n = 4 streams");

constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // Addcmul value = 1.0 (exact)
constexpr uint32_t COEF_POST_BASE = 0;
constexpr uint32_t COEF_COMB_BASE = n;

// Coefficient tile (Col-indexed by output stream j), caller-managed.
constexpr auto coef_in = ckl::input(
    cb_coef_cols,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Col,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Offset);
constexpr auto f_in = ckl::input(
    cb_f_tiles,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Row,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Direct);
constexpr auto x_in = ckl::input(
    cb_x_tiles,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Row,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Offset);
constexpr auto out_spec = ckl::output(cb_out_tiles, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile);

using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, coef_in, ckl::Dst::D1>;
using CopyF = ckl::CopyTile<f_in, ckl::Dst::D2>;
using CopyX = ckl::CopyTile<x_in, ckl::Dst::D2>;
using MulPostF = ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>;
using MacCombX = ckl::Addcmul<DataFormat::Float32, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>;
using PackOut = ckl::PackTile<out_spec, ckl::Dst::D0>;

// mix_block: one chain over grid(n, extent). X block layout in the CB is [i][c] with the
// nominal block_c stride (ragged tail keeps nominal offsets).
ALWI void mix_block(uint32_t extent, DataflowBuffer& coef, DataflowBuffer& f, DataflowBuffer& x, DataflowBuffer& out) {
    ckl::eltwise_chain(
        ckl::IterationShape::grid(n, extent),
        CoefBcast{coef, COEF_POST_BASE},
        CopyF{f},
        MulPostF{},
        CoefBcast{coef, COEF_COMB_BASE + 0 * n},
        CopyX{x, 0 * block_c},
        MacCombX{ONE_F32_BITS},
        CoefBcast{coef, COEF_COMB_BASE + 1 * n},
        CopyX{x, 1 * block_c},
        MacCombX{ONE_F32_BITS},
        CoefBcast{coef, COEF_COMB_BASE + 2 * n},
        CopyX{x, 2 * block_c},
        MacCombX{ONE_F32_BITS},
        CoefBcast{coef, COEF_COMB_BASE + 3 * n},
        CopyX{x, 3 * block_c},
        MacCombX{ONE_F32_BITS},
        PackOut{out});
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_c = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;

    DataflowBuffer coef_buf(cb_coef_cols);
    DataflowBuffer f_buf(cb_f_tiles);
    DataflowBuffer x_buf(cb_x_tiles);
    DataflowBuffer out_buf(cb_out_tiles);

    compute_kernel_hw_startup(cb_f_tiles, cb_x_tiles, cb_out_tiles);

    for (uint32_t r = 0; r < num_rows; ++r) {
        coef_buf.wait_front(n_coef_cols);  // load_coefficients (consumer side)
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
            f_buf.wait_front(block_c);
            x_buf.wait_front(n * block_c);
            mix_block(extent, coef_buf, f_buf, x_buf, out_buf);
            f_buf.pop_front(block_c);
            x_buf.pop_front(n * block_c);
            // Ragged tail: keep the output window nominal (n * block_c pages per block) so the
            // writer's block window never wraps the CB ring. The padding pages are never written.
            if (extent < block_c) {
                out_buf.reserve_back(n * (block_c - extent));
                out_buf.push_back(n * (block_c - extent));
            }
        }
        coef_buf.pop_front(n_coef_cols);  // release_coefficients
    }
}
