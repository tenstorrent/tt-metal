// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute (Phase 0, regime R1 plain).
//
// mix_block over a block of B = block_cols units (column tiles of one token-tile row), per output stream j,
// in ONE dst-sync window (DEST: Dc = coefficient, Do = operand, A_u = accumulator of unit u, B + 2 slots):
//     Dc = bcast_col(post_j)                    ; for u: Do = F[u];   A_u  = Dc * Do        (SFPU fp32)
//     for i: Dc = bcast_col(comb_ij)            ; for u: Do = X_i[u]; A_u += Dc * Do        (SFPU fp32 MAD)
//     for u: pack A_u -> cb_mixed_out_tiles     (output order inside a block: j-major, then u)
// One coefficient broadcast serves the whole block (the reason the block exists: fp32 unpack-to-dest is the
// compute bottleneck). Every coefficient product is fp32 SFPU arithmetic (Addcmul with value 1.0 is a single-
// rounding MAD); the FPU never multiplies a coefficient (tf32 bias). fp32 CBs reach DEST by unpack-to-dest
// (UnpackToDestFp32 set on the host). A ragged last pass of a row runs as blocks of one unit.
//
// CB lifecycles owned by this kernel (the chain inputs are caller-managed: WaitPolicy::None / PopPolicy::None,
// offset-addressed from the CB front; a pass always spans block_cols ring slots, a ragged pass carries filler):
//   cb_coef_cols     : wait n + n^2 pages per token row, popped after the row's last block.
//   cb_sublayer_tiles: wait / pop block_cols per pass.  cb_stream_tiles: wait / pop block_cols*n (unit-major).
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
constexpr uint32_t mix_fixed_slots = get_compile_time_arg_val(7);  // host MIX_FIXED_SLOTS: Dc + Do
constexpr uint32_t n_coef = n + n * n;
constexpr uint32_t ONE_F32 = 0x3f800000u;  // Addcmul value: A + 1.0 * (Dc * Do) == fma(Dc, Do, A)

static_assert(mix_fixed_slots == 2, "mix_block DEST layout: D0 = coefficient, D1 = operand, accumulators after");
static_assert(
    block_cols + mix_fixed_slots <= ckl::DEST_AUTO_LIMIT, "mix_block: block_cols + fixed DEST slots must fit");

constexpr ckl::Dst Dc = ckl::Dst::D0;  // coefficient (column-broadcast)
constexpr ckl::Dst Do = ckl::Dst::D1;  // operand (F or X_i of one unit)
template <size_t U>
constexpr ckl::Dst acc_slot() {
    return static_cast<ckl::Dst>(mix_fixed_slots + U);
}

// Caller-managed, offset-addressed inputs: the page index (from the CB front) is the element's runtime base.
constexpr ckl::InputSpec caller_managed(uint32_t cb) {
    return ckl::input(
        cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}

using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef_cols), Dc>;
using SublayerCopy = ckl::CopyTile<caller_managed(cb_sublayer_tiles), Do>;
using StreamCopy = ckl::CopyTile<caller_managed(cb_stream_tiles), Do>;

// post_j term over the block of units base..base+B-1: A_u = post_j * F[u]
template <size_t... U>
ALWI auto sublayer_term(uint32_t j, uint32_t base, std::index_sequence<U...>) {
    return std::tuple_cat(
        std::make_tuple(CoefBcast{j}),
        std::make_tuple(SublayerCopy{base + U}, ckl::MulBinary<Dc, Do, acc_slot<U>()>{})...);
}

// comb_ij term over the block: A_u += comb_ij * X_i[u]
template <uint32_t I, size_t... U>
ALWI auto stream_term(uint32_t j, uint32_t base, std::index_sequence<U...>) {
    return std::tuple_cat(
        std::make_tuple(CoefBcast{n + I * n + j}),
        std::make_tuple(
            StreamCopy{(base + U) * n + I},
            ckl::Addcmul<DataFormat::Float32, acc_slot<U>(), Dc, Do, acc_slot<U>()>{ONE_F32})...);
}

template <size_t... U>
ALWI auto pack_block(std::index_sequence<U...>) {
    return std::make_tuple(ckl::PackTile<ckl::output(cb_mixed_out_tiles), acc_slot<U>()>{}...);
}

// X'_j of the B units base..base+B-1 of the waited window, packed j-major.
template <uint32_t B, size_t... I>
ALWI void mix_units_j(uint32_t j, uint32_t base, std::index_sequence<I...>) {
    constexpr auto units = std::make_index_sequence<B>{};
    auto elems = std::tuple_cat(sublayer_term(j, base, units), stream_term<I>(j, base, units)..., pack_block(units));
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

template <uint32_t B>
ALWI void mix_units(uint32_t base) {
    for (uint32_t j = 0; j < n; ++j) {
        mix_units_j<B>(j, base, std::make_index_sequence<n>{});
    }
}

}  // namespace

void kernel_main() {
    const uint32_t t_count = get_arg_val<uint32_t>(0);
    const uint32_t c_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_stream_tiles, cb_coef_cols, cb_mixed_out_tiles);

    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            cb_wait_front(cb_coef_cols, n_coef);  // the row's expanded coefficients, held for every block
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                // mix_block: every pass occupies block_cols ring slots (a ragged pass carries filler slots)
                cb_wait_front(cb_sublayer_tiles, block_cols);
                cb_wait_front(cb_stream_tiles, block_cols * n);
                if (cols_in_pass == block_cols) {
                    mix_units<block_cols>(0);
                } else {
                    for (uint32_t u = 0; u < cols_in_pass; ++u) {  // ragged pass: blocks of one unit
                        mix_units<1>(u);
                    }
                }
                cb_pop_front(cb_sublayer_tiles, block_cols);
                cb_pop_front(cb_stream_tiles, block_cols * n);
            }
            cb_pop_front(cb_coef_cols, n_coef);
        }
    }
}
