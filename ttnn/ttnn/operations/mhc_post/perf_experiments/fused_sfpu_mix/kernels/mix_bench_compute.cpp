// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench of mhc_post's mix_block (R2: has_w, packed coefficients, B = 4 units per pass, n = 4).
//
// VARIANT 0 (baseline) — the op's current mix_units_j<B>: per output stream j one eltwise_chain window
//   DEST: D0 = Dc (column-broadcast coefficient), D1 = Do (operand), D2 = Dsq, D3.. = A_u
//   per j: Dc = bcast(post_j); per u: Do = F[u]; A_u = Dc * Do
//          per i: Dc = bcast(comb_ij); per u: Do = X_i[u]; A_u = fma(Dc, Do, A_u)
//          pack A_u -> out, bf16; Dsq = reload; Dsq = fma(A_u, A_u, Dsq) per u; pack Dsq
//   => per unit 20 operand unpack->DEST copies, 24 SFPU tile ops; per pass 20 coefficient broadcasts.
//
// VARIANT 1 (candidate, "operand-outer fused sweep") — one dst-sync window per UNIT:
//   DEST: D0..D3 = A_0..A_3 (the unit's 4 output streams), D4 = Do (operand), D5 = Dk (coefficient page),
//   D6 = Dsq. For s in {F, X0, X1, X2, X3}: Do = operand s (copied ONCE per unit), Dk = coefficient page K_s,
//   then ONE custom SFPU sweep updates all 4 outputs: A_j = c_{s,j} * x (s = F) or A_j = fma(c_{s,j}, x, A_j);
//   the last sweep also folds Dsq = fma(A_j, A_j, Dsq) for j = 0..3. Outputs packed out of order at
//   k = j*B + u so both output CBs keep the op's j-major page order.
//   Coefficient page K_s (fp32, built by the reader / host): element (row R, col C) = coef(s, j)[token R] with
//   j = 2*(C >= 16) + (C & 1) — the SFPU vectors at offsets {0, 1, 8, 9} from a row-group base (even/odd columns of
//   the left / right face) then carry exactly c_{s,0..3} of that row group's 4 tokens; one load serves 4 vectors.
//   => per unit 5 operand copies + 5 coefficient-page copies (UnpackToDestFp32) + 5 SFPU calls.
//   A is bit-identical to the baseline (same fp32 MAD sequence per element). Dsq folds in unit-major order
//   (u outer, j inner) instead of the baseline's j outer, u inner: same terms, different fp32 summation order.
//
// Raw-sfpi justification (helper-first policy): the eltwise_chain SFPU binary/ternary elements (MulBinary /
// Addcmul) each walk ONE output tile and need every operand as a full DEST tile; a mix step that updates four
// output tiles from one operand load and a 4-coefficient page (one coefficient load serving 4 vectors) is not
// expressible as chain elements — it needs one custom SFPU program addressing 4 accumulators + operand + page.
// The copies and packs stay chain elements (same LLKs, inits and reconfigs as the baseline).

#include <cstdint>
#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_stream_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_coef = get_compile_time_arg_val(2);  // V0: 24 column-0 pages; V1: 5 K pages
constexpr uint32_t cb_mixed_out_tiles = get_compile_time_arg_val(3);
constexpr uint32_t cb_mixed_bf16 = get_compile_time_arg_val(4);
constexpr uint32_t cb_sumsq_acc = get_compile_time_arg_val(5);
constexpr uint32_t cb_dsq_out = get_compile_time_arg_val(6);
constexpr uint32_t num_passes = get_compile_time_arg_val(7);
constexpr uint32_t variant = get_compile_time_arg_val(8);
constexpr uint32_t n_coef_pages = get_compile_time_arg_val(9);
constexpr uint32_t block_cols = get_compile_time_arg_val(10);  // V0 only (V1 is per unit)
constexpr uint32_t ablate = get_compile_time_arg_val(11);
constexpr bool has_w = get_compile_time_arg_val(12) != 0;  // 0: plain mode (no Dsq, one output CB)
constexpr uint32_t n = 4;
constexpr uint32_t ONE_F32 = 0x3f800000u;

constexpr ckl::InputSpec caller_managed(uint32_t cb) {
    return ckl::input(
        cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}
constexpr ckl::OutputSpec out_of_order(uint32_t cb) {
    return ckl::output(
        cb,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}

// ======================= VARIANT 0: the op's current mix_units_j (verbatim structure) =======================
constexpr uint32_t mix_fixed_slots = has_w ? 3 : 2;
constexpr ckl::Dst Dc = ckl::Dst::D0;
constexpr ckl::Dst Do = ckl::Dst::D1;
constexpr ckl::Dst Dsq = ckl::Dst::D2;
template <size_t U>
constexpr ckl::Dst acc_slot() {
    return static_cast<ckl::Dst>(mix_fixed_slots + U);
}
using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, caller_managed(cb_coef), Dc>;
using SublayerCopy = ckl::CopyTile<caller_managed(cb_sublayer_tiles), Do>;
using StreamCopy = ckl::CopyTile<caller_managed(cb_stream_tiles), Do>;

template <size_t... U>
ALWI auto sublayer_term(uint32_t j, uint32_t base, std::index_sequence<U...>) {
    return std::tuple_cat(
        std::make_tuple(CoefBcast{j}),
        std::make_tuple(SublayerCopy{base + U}, ckl::MulBinary<Dc, Do, acc_slot<U>()>{})...);
}
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
    if constexpr (has_w) {
        return std::make_tuple(
            ckl::PackTile<ckl::output(cb_mixed_out_tiles), acc_slot<U>()>{}...,
            ckl::PackTile<ckl::output(cb_mixed_bf16), acc_slot<U>()>{}...);
    } else {
        return std::make_tuple(ckl::PackTile<ckl::output(cb_mixed_out_tiles), acc_slot<U>()>{}...);
    }
}
template <size_t... U>
ALWI auto sumsq_term(std::index_sequence<U...>) {
    if constexpr (has_w) {
        return std::make_tuple(
            ckl::CopyTile<ckl::input(cb_sumsq_acc), Dsq>{},
            ckl::Addcmul<DataFormat::Float32, Dsq, acc_slot<U>(), acc_slot<U>(), Dsq>{ONE_F32}...,
            ckl::PackTile<ckl::output(cb_sumsq_acc), Dsq>{});
    } else {
        return std::make_tuple();
    }
}
template <uint32_t B, size_t... I>
ALWI void mix_units_j(uint32_t j, uint32_t base, std::index_sequence<I...>) {
    constexpr auto units = std::make_index_sequence<B>{};
    auto elems = std::tuple_cat(
        sublayer_term(j, base, units), stream_term<I>(j, base, units)..., pack_block(units), sumsq_term(units));
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

// ======================= VARIANT 1..3: operand-outer fused SFPU sweep =======================
// variant 1: K page always in D5. variant 2: K pages alternate D5 / D7 (the unpacker may write the next page while the
// previous sweep still reads the other slot). variant 3: variant 2 + two accumulators in flight (MAD latency hidden).
// variant 4: two accumulators in flight, K page always in D5.
constexpr uint32_t S_A0 = 0;  // A_0..A_3 in D0..D3
constexpr uint32_t S_OP = 4;  // operand
constexpr uint32_t S_SQ = 6;  // Dsq
constexpr bool k_double = variant == 2 || variant == 3;
constexpr bool interleave = variant >= 3;
constexpr uint32_t k_slot(uint32_t s) { return (k_double && (s & 1)) ? 7u : 5u; }

#ifdef TRISC_MATH
namespace fused {
using sfpi::dst_reg;
using sfpi::vFloat;
constexpr uint32_t T = 32;  // sfpi vectors per fp32 DEST tile

// MODE 0: A_j = c_j * x (sublayer term). MODE 1: A_j = fma(c_j, x, A_j). MODE 2: MODE 1 + Dsq = fma(A_j, A_j, Dsq)
// for j = 0..3. The DEST pointer walks the 8 row groups (+= 2 inside a face pair, += 8 between the pairs); the 4
// vectors of a row group sit at {0, 1, 8, 9} from it in every tile, and so do the group's 4 coefficients in K.
template <uint32_t MODE, uint32_t ADDR>
sfpi_inline void acc_step(vFloat c, vFloat x, vFloat& q) {
    vFloat a = dst_reg[ADDR];
    a = c * x + a;
    dst_reg[ADDR] = a;
    if constexpr (MODE == 2) {
        q = a * a + q;
    }
}
template <uint32_t MODE, uint32_t A0, uint32_t A1>
sfpi_inline void acc_pair(vFloat c0, vFloat c1, vFloat x, vFloat& q) {
    vFloat a = dst_reg[A0];
    vFloat b = dst_reg[A1];
    a = c0 * x + a;
    b = c1 * x + b;
    dst_reg[A0] = a;
    dst_reg[A1] = b;
    if constexpr (MODE == 2) {
        q = a * a + q;
        q = b * b + q;
    }
}

template <uint32_t MODE, uint32_t KS, uint32_t OFF>
sfpi_inline void vec_at(vFloat c0, vFloat c1, vFloat c2, vFloat c3) {
    vFloat x = dst_reg[S_OP * T + OFF];
    if constexpr (MODE == 0) {
        dst_reg[0 * T + OFF] = c0 * x;
        dst_reg[1 * T + OFF] = c1 * x;
        dst_reg[2 * T + OFF] = c2 * x;
        dst_reg[3 * T + OFF] = c3 * x;
    } else {
        vFloat q;  // only live in MODE 2
        if constexpr (MODE == 2) {
            q = dst_reg[S_SQ * T + OFF];
        }
        if constexpr (interleave) {
            acc_pair<MODE, 0 * T + OFF, 1 * T + OFF>(c0, c1, x, q);
            acc_pair<MODE, 2 * T + OFF, 3 * T + OFF>(c2, c3, x, q);
        } else {
            acc_step<MODE, 0 * T + OFF>(c0, x, q);
            acc_step<MODE, 1 * T + OFF>(c1, x, q);
            acc_step<MODE, 2 * T + OFF>(c2, x, q);
            acc_step<MODE, 3 * T + OFF>(c3, x, q);
        }
        if constexpr (MODE == 2) {
            dst_reg[S_SQ * T + OFF] = q;
        }
    }
}

template <uint32_t MODE, uint32_t KS>
inline void sweep() {
#pragma GCC unroll 1
    for (uint32_t h = 0; h < 2; ++h) {
#pragma GCC unroll 1
        for (uint32_t rg = 0; rg < 4; ++rg) {
            vFloat c0 = dst_reg[KS * T + 0];
            vFloat c1 = dst_reg[KS * T + 1];
            vFloat c2 = dst_reg[KS * T + 8];
            vFloat c3 = dst_reg[KS * T + 9];
            vec_at<MODE, KS, 0>(c0, c1, c2, c3);
            vec_at<MODE, KS, 1>(c0, c1, c2, c3);
            vec_at<MODE, KS, 8>(c0, c1, c2, c3);
            vec_at<MODE, KS, 9>(c0, c1, c2, c3);
            dst_reg += 2;
        }
        dst_reg += 3;
        dst_reg += 3;
        dst_reg += 2;
    }
}
}  // namespace fused
#endif

template <uint32_t MODE, uint32_t KS>
struct MixSweep : ckl::UnaryOp<MixSweep<MODE, KS>, static_cast<ckl::Dst>(KS > S_SQ ? KS : S_SQ)> {
    static ALWI void init() { MATH((SFPU_UNARY_INIT(unused))); }
    static ALWI void exec_impl(uint32_t /*slot_offset*/) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(fused::sweep<MODE, KS>, 0, VectorMode::RC_custom)));
    }
};

using OpF = ckl::CopyTile<caller_managed(cb_sublayer_tiles), static_cast<ckl::Dst>(S_OP)>;
using OpX = ckl::CopyTile<caller_managed(cb_stream_tiles), static_cast<ckl::Dst>(S_OP)>;
template <uint32_t S>
using KPage = ckl::CopyTile<caller_managed(cb_coef), static_cast<ckl::Dst>(k_slot(S))>;
template <uint32_t J>
using PackOut = ckl::PackTile<out_of_order(cb_mixed_out_tiles), static_cast<ckl::Dst>(S_A0 + J)>;
template <uint32_t J>
using PackBf = ckl::PackTile<out_of_order(cb_mixed_bf16), static_cast<ckl::Dst>(S_A0 + J)>;

// ablate (bench-only CT arg): bit 0 = drop the SFPU sweeps, bit 1 = drop the operand copies, bit 2 = drop the K copies
template <uint32_t MODE, uint32_t S>
ALWI auto sweep_elem() {
    if constexpr (ablate % 2 == 1) {
        return std::make_tuple();
    } else {
        return std::make_tuple(MixSweep<MODE, k_slot(S)>{});
    }
}
template <class OpCopy, uint32_t S>
ALWI auto copies(uint32_t op_idx) {
    constexpr bool no_op = (ablate / 2) % 2 == 1, no_k = (ablate / 4) % 2 == 1;
    if constexpr (no_op && no_k) {
        return std::make_tuple();
    } else if constexpr (no_op) {
        return std::make_tuple(KPage<S>{S});
    } else if constexpr (no_k) {
        return std::make_tuple(OpCopy{op_idx});
    } else {
        return std::make_tuple(OpCopy{op_idx}, KPage<S>{S});
    }
}

ALWI auto sq_reload() {
    if constexpr (has_w) {
        return std::make_tuple(ckl::CopyTile<ckl::input(cb_sumsq_acc), static_cast<ckl::Dst>(S_SQ)>{});
    } else {
        return std::make_tuple();
    }
}
ALWI auto fused_packs(uint32_t B, uint32_t u) {
    if constexpr (has_w) {
        return std::make_tuple(
            PackOut<0>{0 * B + u},
            PackOut<1>{1 * B + u},
            PackOut<2>{2 * B + u},
            PackOut<3>{3 * B + u},
            PackBf<0>{0 * B + u},
            PackBf<1>{1 * B + u},
            PackBf<2>{2 * B + u},
            PackBf<3>{3 * B + u},
            ckl::PackTile<ckl::output(cb_sumsq_acc), static_cast<ckl::Dst>(S_SQ)>{});
    } else {
        return std::make_tuple(
            PackOut<0>{0 * B + u}, PackOut<1>{1 * B + u}, PackOut<2>{2 * B + u}, PackOut<3>{3 * B + u});
    }
}

// one unit: g = global unit index (F tile g, X tiles g*n + i), u = unit within the pass (pack index j*B + u).
// Dsq is reloaded mid-window right before the last sweep (as the baseline reloads it after the mix terms).
ALWI void fused_unit(uint32_t g, uint32_t u, uint32_t B) {
    auto elems = std::tuple_cat(
        copies<OpF, 0>(g),
        sweep_elem<0, 0>(),
        copies<OpX, 1>(g * n + 0),
        sweep_elem<1, 1>(),
        copies<OpX, 2>(g * n + 1),
        sweep_elem<1, 2>(),
        copies<OpX, 3>(g * n + 2),
        sweep_elem<1, 3>(),
        sq_reload(),
        copies<OpX, 4>(g * n + 3),
        sweep_elem<has_w ? 2 : 1, 4>(),
        fused_packs(B, u));
    std::apply([](auto... e) { ckl::eltwise_chain(ckl::IterationShape::one_tile(), e...); }, elems);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t units = num_passes * block_cols;
    compute_kernel_hw_startup(cb_stream_tiles, cb_coef, cb_mixed_out_tiles);

    // sharded inputs are resident: publish them once
    cb_reserve_back(cb_sublayer_tiles, units);
    cb_push_back(cb_sublayer_tiles, units);
    cb_reserve_back(cb_stream_tiles, units * n);
    cb_push_back(cb_stream_tiles, units * n);
    cb_reserve_back(cb_coef, n_coef_pages);
    cb_push_back(cb_coef, n_coef_pages);
    cb_wait_front(cb_sublayer_tiles, units);
    cb_wait_front(cb_stream_tiles, units * n);
    cb_wait_front(cb_coef, n_coef_pages);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::FillScalar<ckl::Dst::D0>{0.0f},
        ckl::PackTile<ckl::output(cb_sumsq_acc), ckl::Dst::D0>{});

    {
        DeviceZoneScopedN("MIX");
        for (uint32_t p = 0; p < num_passes; ++p) {
            if constexpr (variant == 0) {
                for (uint32_t j = 0; j < n; ++j) {
                    mix_units_j<block_cols>(j, p * block_cols, std::make_index_sequence<n>{});
                }
            } else {
                cb_reserve_back(cb_mixed_out_tiles, n * block_cols);
                if constexpr (has_w) {
                    cb_reserve_back(cb_mixed_bf16, n * block_cols);
                }
                for (uint32_t u = 0; u < block_cols; ++u) {
                    fused_unit(p * block_cols + u, u, block_cols);
                }
                cb_push_back(cb_mixed_out_tiles, n * block_cols);
                if constexpr (has_w) {
                    cb_push_back(cb_mixed_bf16, n * block_cols);
                }
            }
        }
    }

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<ckl::input(cb_sumsq_acc), ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_dsq_out), ckl::Dst::D0>{});
}
