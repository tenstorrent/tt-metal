// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute (Phase 0, regime R1 plain).
//
// mix_block: for every unit (token-tile row t, column tile c) and output stream j, one eltwise_chain in one
// dst-sync window (3 DEST slots):
//     D0 = bcast_col(post_j) ; D1 = F ; D0 = D0 * D1                       (SFPU fp32)
//     for i: D1 = bcast_col(comb_ij) ; D2 = X_i ; D1 = D1 * D2 ; D0 += D1   (SFPU fp32)
//     pack D0 -> cb_mixed_out_tiles
// Every coefficient product is fp32 SFPU lane arithmetic; the FPU never multiplies a coefficient (tf32 bias).
// fp32 CBs reach DEST through unpack-to-dest (UnpackToDestFp32 set on the host).
//
// CB lifecycles owned by this kernel (the chain elements are caller-managed: WaitPolicy::None / PopPolicy::None):
//   cb_coef_cols     : wait n + n^2 pages per token row, popped after the row's last unit.
//   cb_sublayer_tiles: wait 1 / pop 1 per unit.     cb_stream_tiles: wait n / pop n per unit.
// The output CB is chain-owned (PackTile reserve/push per tile).

#include <cstdint>
#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_stream_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(2);
constexpr uint32_t cb_mixed_out_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_rows = get_compile_time_arg_val(5);
constexpr uint32_t block_cols = get_compile_time_arg_val(6);
constexpr uint32_t n_coef = n + n * n;

// Caller-managed, offset-addressed inputs: the page index is the element's runtime base.
constexpr ckl::InputSpec caller_managed(uint32_t cb) {
    return ckl::input(
        cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}

template <ckl::Dst D>
using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), D>;
using SublayerCopy = ckl::CopyTile<caller_managed(cb_sublayer_tiles), ckl::Dst::D1>;
using StreamCopy = ckl::CopyTile<caller_managed(cb_stream_tiles), ckl::Dst::D2>;
using MixedPack = ckl::PackTile<ckl::output(cb_mixed_out_tiles), ckl::Dst::D0>;

// One stream term: D1 = comb_ij (col bcast); D2 = X_i; D1 = D1 * D2; D0 = D0 + D1.
template <uint32_t I>
ALWI auto stream_term(uint32_t j) {
    return std::make_tuple(
        CoefBcast<ckl::Dst::D1>{n + I * n + j},
        StreamCopy{I},
        ckl::Addcmul<DataFormat::Float32, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{0x3f800000u});
}

// X'_j of the current unit, packed to cb_mixed_out_tiles.
template <size_t... I>
ALWI void mix_tile(uint32_t j, std::index_sequence<I...>) {
    auto elems = std::tuple_cat(
        std::make_tuple(
            CoefBcast<ckl::Dst::D0>{j}, SublayerCopy{0}, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{}),
        stream_term<I>(j)...,
        std::make_tuple(MixedPack{}));
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

}  // namespace

void kernel_main() {
    const uint32_t t_count = get_arg_val<uint32_t>(0);
    const uint32_t c_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_stream_tiles, cb_coef_cols, cb_mixed_out_tiles);

    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            cb_wait_front(cb_coef_cols, n_coef);  // the row's expanded coefficients, held for every unit
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                // mix_block over this pass
                for (uint32_t u = 0; u < cols_in_pass; ++u) {
                    cb_wait_front(cb_sublayer_tiles, 1);
                    cb_wait_front(cb_stream_tiles, n);
                    for (uint32_t j = 0; j < n; ++j) {
                        mix_tile(j, std::make_index_sequence<n>{});
                    }
                    cb_pop_front(cb_sublayer_tiles, 1);
                    cb_pop_front(cb_stream_tiles, n);
                }
            }
            cb_pop_front(cb_coef_cols, n_coef);
        }
    }
}
