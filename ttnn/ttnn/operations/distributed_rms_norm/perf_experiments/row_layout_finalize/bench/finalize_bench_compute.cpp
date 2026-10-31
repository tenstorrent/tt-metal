// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated micro-bench of distributed_rms_norm's root finalize (one wave of n tile-rows, G fp32 slot records each ->
// inv_rms = rsqrt(sum_q(slot q) * inv_w + eps)), measured as a serialized latency loop:
//   per rep: reserve out, finalize chain (copies + SFPU + pack), push out, [unpack thread] wait out, pop out
// so the next rep's unpack cannot start before this rep's pack is done (latency, not throughput).
//
// Variants (CT arg 0):
//   0 base       the op's current chain (FINALIZE_ONE_PASS): G slot tiles, record in column 0; G CopyTile ->
//                D0..D(G-1), SumSlotsCol (one sfpi pass, VectorMode::C), MulUnary / AddUnary / Rsqrt as
//                VectorMode::C chain elements (16 vectors each), pack.
//   1 col_fused  same input layout (column 0 of G tiles); G CopyTile, then ONE fused sfpi element (fold + scale + eps
//                + rsqrt) over the 8 even-parity vectors of faces 0 / 2 (column 0 lives only there).
//   2 row_fused  G slot tiles, record in face-0 rows 0-1 (32 contiguous words); G CopyTile, fused element over the 2
//                vectors covering face rows 0-3 (both parities).
//   3 packed     ONE tile per tile-row: slot q's record in face (q/4), rows 4(q%4) .. +1 (32 contiguous words each);
//                1 CopyTile, fused element over 2 vectors.
//   4 dense      ONE tile per tile-row: slot q's record fills exactly SFPU vector q (rows 4(q/2)..+3, columns of
//                parity q%2); 1 CopyTile, fused element over 1 vector.
//   5 floor_G    G CopyTile + pack (no math): the unpack / pack floor of the G-tile layouts.
//   6 floor_1    1 CopyTile + pack (no math): the floor of the one-tile layouts.
// Raw LLK (bench and candidate): the fused element calls _llk_math_eltwise_unary_sfpu_params_ with a hand-written sfpi
// body; no kernel_lib element folds N DEST tiles + scale + eps + rsqrt on a chosen vector subset (the kernel_lib
// elements are whole-tile VectorMode::RC, one op per element, each with its own SFPU init).

#include <stdint.h>

#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

namespace ckl = compute_kernel_lib;

// ---------------- baseline: the op's column elements, verbatim ----------------
namespace ckernel {
template <int OP>
ALWI void drms_binop_scalar_tile_col(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        param1));
}
#ifdef TRISC_MATH
namespace sfpu {
template <int ITERATIONS, uint32_t N>
inline void drms_calculate_sum_slots(const uint32_t d0, const uint32_t /*d_last*/, const uint32_t d_out) {
    constexpr uint32_t dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat acc = sfpi::dst_reg[d0 * dst_tile_size_sfpi];
#pragma GCC unroll 8
        for (uint32_t q = 1; q < N; ++q) {
            acc = acc + sfpi::dst_reg[(d0 + q) * dst_tile_size_sfpi];
        }
        sfpi::dst_reg[d_out * dst_tile_size_sfpi] = acc;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
#endif
template <uint32_t N>
ALWI void drms_sum_slots_tile_col(uint32_t idst0, uint32_t odst) {
    MATH(
        (_sfpu_binary_check_<DST_SYNC_MODE>(idst0, idst0 + N - 1, odst, VectorMode::C),
         _llk_math_eltwise_binary_sfpu_params_(
             ::ckernel::sfpu::drms_calculate_sum_slots<8, N>, idst0, idst0 + N - 1, odst, VectorMode::C)));
}
ALWI void drms_rsqrt_tile_col(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false),
        idst,
        VectorMode::C));
}

// ---------------- candidate: one fused sfpi pass over a chosen vector subset ----------------
// sfpi unit u of a DEST tile = one 32-lane vector: face u/8, rows 4*((u%8)/2) .. +3, columns of parity u%2.
// L (layout): 1 col (G tiles, col 0), 2 row (G tiles, face-0 rows 0-1), 3 packed (1 tile, slot q at face q/4 rows
// 4(q%4)..+1), 4 dense (1 tile, slot q = vector q).
template <uint32_t L>
constexpr uint32_t fin_nvec() {
    return L == 1 ? 8 : (L == 4 ? 1 : 2);
}
template <uint32_t L>
constexpr uint32_t fin_vec_unit(uint32_t v) {
    return L == 1 ? (v / 4) * 16 + (v % 4) * 2 : v;  // col: even units of faces 0 / 2; else units 0 (,1)
}
template <uint32_t L>
constexpr uint32_t fin_slot_off(uint32_t q) {
    return (L == 1 || L == 2) ? q * 32 : (L == 3 ? (q / 4) * 8 + (q % 4) * 2 : q);
}
#ifdef TRISC_MATH
namespace sfpu {
// Same SFPU ops, same order, same DEST round trips as the baseline's four passes: the fold accumulates in a register
// and is stored (SumSlotsCol), then each stage reloads its input from DEST and stores its result (binop_with_scalar
// MUL, ADD, rsqrt): the store / reload pairs keep the per-stage DEST rounding of 16-bit DEST and stop the compiler from
// fusing x*inv_w + eps into one SFPMAD. The result lands in the slot-0 position of each processed vector.
template <uint32_t L, uint32_t u, uint32_t... Q>
sfpi_inline void drms_finalize_vec(
    const sfpi::vFloat inv_w, const sfpi::vFloat eps, std::integer_sequence<uint32_t, Q...>) {
    sfpi::vFloat acc = sfpi::dst_reg[u];
    ((acc = acc + sfpi::dst_reg[u + fin_slot_off<L>(Q + 1)]), ...);  // left fold, slot order 1..G-1
    sfpi::dst_reg[u] = acc;
    sfpi::vFloat x = sfpi::dst_reg[u];
    x = x * inv_w;
    sfpi::dst_reg[u] = x;
    x = sfpi::dst_reg[u];
    x = x + eps;
    sfpi::dst_reg[u] = x;
    x = sfpi::dst_reg[u];
    sfpi::vFloat y = _calculate_sqrt_body_<APPROX, true, false>(x);
    if constexpr (!DST_ACCUM_MODE) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[u] = y;
}
template <uint32_t L, uint32_t G, uint32_t... V>
sfpi_inline void drms_finalize_all(
    const sfpi::vFloat inv_w, const sfpi::vFloat eps, std::integer_sequence<uint32_t, V...>) {
    (drms_finalize_vec<L, fin_vec_unit<L>(V)>(inv_w, eps, std::make_integer_sequence<uint32_t, G - 1>{}), ...);
}
template <uint32_t L, uint32_t G>
inline void drms_finalize_fused(const uint32_t inv_w_bits, const uint32_t eps_bits) {
    const sfpi::vFloat inv_w = Converter::as_float(inv_w_bits);
    const sfpi::vFloat eps = Converter::as_float(eps_bits);
    drms_finalize_all<L, G>(inv_w, eps, std::make_integer_sequence<uint32_t, fin_nvec<L>()>{});
}
}  // namespace sfpu
#endif
template <uint32_t L, uint32_t G>
ALWI void drms_finalize_fused_tile(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ::ckernel::sfpu::drms_finalize_fused<L, G>, idst, VectorMode::None, inv_w_bits, eps_bits)));
}
}  // namespace ckernel

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 16;

template <uint32_t N, ckl::Dst Last, ckl::Dst Out>
struct SumSlotsCol : ckl::BinaryOp<SumSlotsCol<N, Last, Out>, ckl::Dst::D0, Last, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t off) { ckernel::drms_sum_slots_tile_col<N>(off, ckl::to_u32(Out) + off); }
};
template <int OP, ckl::Dst Slot>
struct BinopScalarCol : ckl::UnaryOp<BinopScalarCol<OP, Slot>, Slot> {
    uint32_t param0;
    constexpr explicit BinopScalarCol(uint32_t p) noexcept : param0(p) {}
    constexpr BinopScalarCol() noexcept : param0(0) {}
    static ALWI void init() { binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t off) const {
        ckernel::drms_binop_scalar_tile_col<OP>(ckl::to_u32(Slot) + off, param0);
    }
};
template <ckl::Dst Slot>
struct RsqrtCol : ckl::UnaryOp<RsqrtCol<Slot>, Slot> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t off) { ckernel::drms_rsqrt_tile_col(ckl::to_u32(Slot) + off); }
};
// Fused finalize element; advertises the last DEST tile it reads (Last) for the chain's DEST accounting.
template <uint32_t L, uint32_t G, ckl::Dst Last>
struct FinalizeFused : ckl::UnaryOp<FinalizeFused<L, G, Last>, Last> {
    uint32_t inv_w_bits, eps_bits;
    constexpr FinalizeFused(uint32_t a, uint32_t b) noexcept : inv_w_bits(a), eps_bits(b) {}
    constexpr FinalizeFused() noexcept : inv_w_bits(0), eps_bits(0) {}
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t off) const {
        ckernel::drms_finalize_fused_tile<L, G>(off, inv_w_bits, eps_bits);
    }
};

static constexpr auto slot_in = ckl::input(
    cb_in,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Col,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Strided);
using OutPack = ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>;

template <uint32_t V, uint32_t G, uint32_t... Q>
ALWI void finalize(uint32_t n, uint32_t inv_w_bits, uint32_t eps_bits, std::integer_sequence<uint32_t, Q...>) {
    if constexpr (V == 0) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
            ckl::CopyTile<slot_in, static_cast<ckl::Dst>(Q + 1)>{ckl::StridedTileRange{Q + 1, G}}...,
            SumSlotsCol<G, static_cast<ckl::Dst>(G - 1), ckl::Dst::D0>{},
            BinopScalarCol<ckernel::MUL_UNARY, ckl::Dst::D0>{inv_w_bits},
            BinopScalarCol<ckernel::ADD_UNARY, ckl::Dst::D0>{eps_bits},
            RsqrtCol<ckl::Dst::D0>{},
            OutPack{});
    } else if constexpr (V == 1 || V == 2) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
            ckl::CopyTile<slot_in, static_cast<ckl::Dst>(Q + 1)>{ckl::StridedTileRange{Q + 1, G}}...,
            FinalizeFused<V, G, static_cast<ckl::Dst>(G - 1)>{inv_w_bits, eps_bits},
            OutPack{});
    } else if constexpr (V == 3 || V == 4) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, 1}},
            FinalizeFused<V, G, ckl::Dst::D0>{inv_w_bits, eps_bits},
            OutPack{});
    } else if constexpr (V == 5) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
            ckl::CopyTile<slot_in, static_cast<ckl::Dst>(Q + 1)>{ckl::StridedTileRange{Q + 1, G}}...,
            OutPack{});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, 1}},
            OutPack{});
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t V = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_in = get_compile_time_arg_val(3);
    const uint32_t reps = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    cb_reserve_back(cb_in, tiles_in);
    cb_push_back(cb_in, tiles_in);  // sharded input resident
    cb_wait_front(cb_in, tiles_in);

    // rep 0 (untimed): the result the host checks. The output CB is the sharded output (n pages).
    finalize<V, G>(n, inv_w_bits, eps_bits, std::make_integer_sequence<uint32_t, G - 1>{});
    cb_wait_front(cb_out, n);
    cb_pop_front(cb_out, n);
    {
        DeviceZoneScopedN("FIN");
        for (uint32_t r = 0; r < reps; ++r) {
            finalize<V, G>(n, inv_w_bits, eps_bits, std::make_integer_sequence<uint32_t, G - 1>{});
            cb_wait_front(cb_out, n);  // unpack thread: this rep's pack is done before the next rep's unpack
            cb_pop_front(cb_out, n);
        }
    }
}
