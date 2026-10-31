// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm compute (op_design.md -> Block schedule / Block Operation Realization / API Mapping).
//
// Per block of R_b tile-rows of this core's W-slice; each block is cut into waves of Rw rows (the exchange unit):
//   per wave k (rows [k*Rw, +n)):
//     square_accumulate   eltwise_chain: x*x DEST-accumulated per tile-row over the slice's valid tiles
//                         -> cb_sumsq_partial (raw lanes)
//     collapse            reduce<SUM, REDUCE_ROW>(n x 1), scaler 1.0, NO post-op -> column-0 partial
//                           R3 (G_w > 1): cb_partial_col (the reader moves it to the root's cb_gather)
//                           R1 / R1-HS:   cb_dev_partial (this core is its own root)
//                         masked core (sharded, W % 32 != 0): the partial last tile reduced with the PARTIAL scaler,
//                         accumulated onto the full tiles' collapse (base op W-pad mask)
//     combine (R3 root)   reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>(G_w x n, row stride Rw) over the gathered
//                         member partials, NO post-op -> cb_dev_partial (the device partial; the reader publishes it)
//   scale_by_gamma chunk 0 of the block (hoisted: needs no remote data, it is the exchange's stall shadow)
//   per chunk c, per wave k:
//     c == 0, root: finalize_wave  eltwise_chain over the wave's G gathered slot tiles (cb_stats_gather, fp32,
//                         UnpackToDestFp32): slot 0 -> D0, then for q = 1..G-1 slot q -> D1, D0 += D1 (fixed slot
//                         order on every chip -> bit-identical ss), x(1/W_global), +eps, rsqrt
//                         -> cb_rms_root (R3: the reader distributes it) / cb_inv_rms (R1: this core is the group)
//     c == 0:             wait cb_inv_rms cumulatively through wave k
//     scale_by_gamma (c > 0, whole block, before the waves of chunk c) and scale_by_inv_rms of wave k's rows:
//                         (x*gamma | x) * inv_rms (Col bcast) -> cb_output (interleaved: C pages per tile-row;
//                         sharded: packed strided into the output shard)
//   release: pop cb_x_tiles (R_b * WS), cb_inv_rms (R_b)
// Every phase is a kernel_lib helper; the raw calls are the caller-owned CB waits / pops the caller-managed
// (None / None) policies hand back.
// DEVIATION (Refinement 3): finalize_wave's SFPU elements are in-kernel chain elements (AddBinaryCol, BinopScalarCol,
// RsqrtCol) issuing the API's own LLK calls with VectorMode::C (faces 0 + 2): the kernel_lib elements hardcode
// VectorMode::RC and the finalize tiles are valid in column 0 only (measured 21.05 -> 20.51 us at 640x1792).
// DEVIATION (Refinement 6): SumSlotsCol folds the G slots (copied into D0..D(G-1)) in ONE sfpi pass instead of G-1
// AddBinaryCol elements: no kernel_lib element sums N DEST tiles, and each copy / add pair re-inits the math MOP and
// the SFPU on the first-inv critical path (finalize ablation: ~0.8 us). Same adds, same order: bit-identical.
// DEVIATION (perf experiment row_layout_finalize, host FIN_LAYOUT, CT arg 15): FinalizeFused runs the whole finalize
// (slot fold + x(1/W_global) + eps + rsqrt) as ONE hand-written sfpi pass over only the SFPU vectors that carry data,
// with one SFPU init (rsqrt's) instead of four. Layout 4 (dense): the reader scatters slot q's 32-word record
// into SFPU vector q of ONE tile per tile-row (face q/8, rows 4((q%8)/2)..+3, columns of parity q%2), so the finalize
// is 1 CopyTile + 1 vector of SFPU work per tile-row (vs G CopyTile + 4 x 16 vectors); inv_rms lands in vector 0 of
// cb_rms_root and the root reader re-lays it into column 0 of cb_inv_rms (R1 roots included). Layout 3 (packed): slot
// q in face q/4 rows 4(q%4)..+1 (contiguous), 2 vectors; layout 5 (packed_direct, default) is layout 3 here (the
// reader NoC-reads fp32 records straight into those rows). Layout 1 (col): the column-0 layout kept, the fused pass on
// its 8 even-parity vectors. Bypassed: kernel_lib's AddBinary / MulUnary / AddUnary / Rsqrt elements (whole-tile
// VectorMode::RC, one op and one SFPU init per element; no element folds N DEST tiles or addresses a vector subset).
// Same SFPU ops in the same order with the same DEST store / reload between stages: bit-identical inv_rms (measured).

#include <stdint.h>

#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

// Measurement-only per-stage zones (host DRMS_ZONES=1 -> define; never set in production).
#ifdef DRMS_ZONES
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

namespace ckl = compute_kernel_lib;

// ---- Column-vector SFPU tile ops for finalize_wave (raw LLK, documented deviation) ----
// The kernel_lib SFPU elements (AddBinary, MulUnary, AddUnary, Rsqrt) run VectorMode::RC (all 4 faces). The finalize
// tiles carry data in column 0 only (the records are scattered into column 0; the inv_rms consumers are the Col
// broadcast and the faces-0/2 distribution), so faces 1 / 3 are don't-care. These are the same LLK calls as the API
// (eltwise_binary_sfpu.h, binop_with_scalar.h, rsqrt.h) with VectorMode::C: faces 0 + 2, half the SFPU passes, on the
// exchange's critical tail (collect -> finalize -> distribute). The inits are the API's own (*_tile_init).
namespace ckernel {
ALWI void drms_add_binary_tile_col(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8 /* ITERATIONS */, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        idst0,
        idst1,
        odst,
        VectorMode::C)));
}
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
// One-pass slot fold (Refinement 6, host FINALIZE_ONE_PASS): DEST tiles d0 .. d0+N-1 hold the N gathered slots;
// faces 0 / 2 of tile d0 <- ((s0 + s1) + s2) + ... in slot order. The same SFPU adds, in the same order, as the
// AddBinaryCol fold (calculate_sfpu_binary ADD = in0 + in1), so the sum is bit-identical; it replaces N-1 copy / add
// element pairs (each a math-MOP <-> SFPU re-init on the exchange's critical tail) by N copies + one SFPU pass.
#ifdef TRISC_MATH
namespace sfpu {
template <int ITERATIONS, uint32_t N>
inline void drms_calculate_sum_slots(const uint32_t d0, const uint32_t /*d_last*/, const uint32_t d_out) {
    constexpr uint32_t dst_tile_size_sfpi = 32;  // rows of a DEST tile as seen by sfpi load / store
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

// ---- FinalizeFused (FIN_LAYOUT 1 / 3 / 4) ----
// sfpi unit u of a DEST tile = one 32-lane vector: face u/8, rows 4*((u%8)/2) .. +3, columns of parity u%2.
template <uint32_t L>
constexpr uint32_t fin_nvec() {
    return L == 1 ? 8 : (L == 4 ? 1 : 2);
}
template <uint32_t L>
constexpr uint32_t fin_vec_unit(uint32_t v) {
    return L == 1 ? (v / 4) * 16 + (v % 4) * 2 : v;  // col: the even units of faces 0 / 2; else units 0 (, 1)
}
template <uint32_t L>
constexpr uint32_t fin_slot_off(uint32_t q) {
    return L == 1 ? q * 32 : (L == 3 ? (q / 4) * 8 + (q % 4) * 2 : q);
}
#ifdef TRISC_MATH
namespace sfpu {
// The baseline's four passes per vector, in order: fold (register accumulate, store), x inv_w (reload, store), + eps
// (reload, store), rsqrt (reload, [bf16 RNE under 16-bit DEST], store). The store / reload pairs keep the per-stage
// DEST rounding of 16-bit DEST and stop the compiler from fusing x*inv_w + eps into one SFPMAD.
template <uint32_t L, uint32_t u, uint32_t... Q>
sfpi_inline void drms_finalize_vec(
    const sfpi::vFloat inv_w, const sfpi::vFloat eps, std::integer_sequence<uint32_t, Q...>) {
    sfpi::vFloat acc = sfpi::dst_reg[u];
    ((acc = acc + sfpi::dst_reg[u + fin_slot_off<L>(Q + 1)]), ...);  // left fold, slots 1 .. G-1 in order
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

constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq_partial = 3;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_x_gamma = 7;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_sumsq_last = 12;
constexpr uint32_t cb_sumsq_acc = 13;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_dev_partial = 17;
constexpr uint32_t cb_stats_gather = 18;

template <ckl::Dst In0, ckl::Dst In1, ckl::Dst Out>
struct AddBinaryCol : ckl::BinaryOp<AddBinaryCol<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t off) {
        ckernel::drms_add_binary_tile_col(ckl::to_u32(In0) + off, ckl::to_u32(In1) + off, ckl::to_u32(Out) + off);
    }
};
// N slots in D0 .. D(N-1) folded into Out (faces 0 / 2), one SFPU pass. Advertises D(N-1) as its second input so
// the chain's DEST-capacity accounting sees every slot it reads.
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

// The fused finalize (FIN_LAYOUT 1 / 3 / 4); advertises the last DEST tile it reads for the chain's DEST accounting.
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

// finalize_wave: the cross-device sum of the G slots of each of the wave's n tile-rows (slot q of tile-row t is tile
// t*G + q of the waited window), then inv_rms = rsqrt(ss * (1/W_global) + eps). Left fold over the slots: 2 DEST
// tiles live (D0 running sum, D1 incoming slot), whatever G is.
template <uint32_t G, uint32_t cb_out, uint32_t... Q>
ALWI void finalize_wave_impl(
    uint32_t n, uint32_t inv_w_bits, uint32_t eps_bits, std::integer_sequence<uint32_t, Q...>) {
    static constexpr auto slot_in = ckl::input(
        cb_stats_gather,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr uint32_t L = get_compile_time_arg_val(15) == 5 ? 3 : get_compile_time_arg_val(15);  // FIN_LAYOUT
    constexpr uint32_t TPR = (L == 3 || L == 4) ? 1 : G;  // cb_stats_gather tiles per tile-row
#ifdef DRMS_ABLATE_FINALIZE
    // measurement only: slot 0 straight to the output (sync scaffolding kept, math stubbed)
    ckl::eltwise_chain(
        ckl::IterationShape::grid(n, 1),
        ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, TPR}},
        ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
    return;
#endif
    if constexpr (L == 3 || L == 4) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, 1}},
            FinalizeFused<L, G, ckl::Dst::D0>{inv_w_bits, eps_bits},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
        return;
    } else if constexpr (L == 1 && G <= ckl::DEST_AUTO_LIMIT) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, 1),
            ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
            ckl::CopyTile<slot_in, static_cast<ckl::Dst>(Q + 1)>{ckl::StridedTileRange{Q + 1, G}}...,
            FinalizeFused<1, G, static_cast<ckl::Dst>(G - 1)>{inv_w_bits, eps_bits},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
        return;
    }
    // Host FINALIZE_ONE_PASS (CT arg 14): every slot copied into its own DEST tile, then one SFPU fold pass; needs G
    // DEST tiles (else the 2-tile left fold below, whatever G is).
    constexpr bool one_pass = get_compile_time_arg_val(14) != 0 && G >= 2 && G <= ckl::DEST_AUTO_LIMIT;
    auto fold = [&]() {
        if constexpr (one_pass) {
            return std::make_tuple(
                ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
                ckl::CopyTile<slot_in, static_cast<ckl::Dst>(Q + 1)>{ckl::StridedTileRange{Q + 1, G}}...,
                SumSlotsCol<G, static_cast<ckl::Dst>(G - 1), ckl::Dst::D0>{});
        } else {
            return std::tuple_cat(
                std::make_tuple(ckl::CopyTile<slot_in, ckl::Dst::D0>{ckl::StridedTileRange{0, G}}),
                std::make_tuple(
                    ckl::CopyTile<slot_in, ckl::Dst::D1>{ckl::StridedTileRange{Q + 1, G}},
                    AddBinaryCol<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{})...);
        }
    };
    auto elems = std::tuple_cat(
        fold(),
        std::make_tuple(
            BinopScalarCol<ckernel::MUL_UNARY, ckl::Dst::D0>{inv_w_bits},
            BinopScalarCol<ckernel::ADD_UNARY, ckl::Dst::D0>{eps_bits},
            RsqrtCol<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{}));
    std::apply([&](auto... e) { ckl::eltwise_chain(ckl::IterationShape::grid(n, 1), e...); }, elems);
}

template <uint32_t G, uint32_t cb_out>
ALWI void finalize_wave(uint32_t n, uint32_t inv_w_bits, uint32_t eps_bits) {
    constexpr uint32_t L = get_compile_time_arg_val(15) == 5 ? 3 : get_compile_time_arg_val(15);
    constexpr uint32_t TPR = (L == 3 || L == 4) ? 1 : G;
    cb_wait_front(cb_stats_gather, TPR * n);
    finalize_wave_impl<G, cb_out>(n, inv_w_bits, eps_bits, std::make_integer_sequence<uint32_t, G - 1>{});
    cb_pop_front(cb_stats_gather, TPR * n);
}

template <
    bool HAS_GAMMA,
    uint32_t R,
    uint32_t C,
    uint32_t NUM_W_CHUNKS,
    uint32_t WS,
    bool SHARDED,
    bool COMBINE,
    bool MASKED_CFG,
    uint32_t G_W,
    uint32_t SCALE_BLOCK,
    uint32_t RW,
    uint32_t G,
    bool GAMMA_RESIDENT>
FORCE_INLINE void run(
    uint32_t rows_this_core,
    uint32_t inv_w_bits,
    uint32_t eps_bits,
    uint32_t slice_valid,
    bool is_root,
    bool masked_core) {
    constexpr uint32_t Wp = NUM_W_CHUNKS * C;
    // Local collapse target: the member partial (R3) or the device partial (this core is its group).
    constexpr uint32_t cb_collapsed = COMBINE ? cb_partial_col : cb_dev_partial;
    // Finalize target: R3 root -> cb_rms_root (distributed by the reader); R1 -> this core's cb_inv_rms.
    // FIN_LAYOUT 3 / 4: inv_rms is not in column 0, every root (R1 too) hands it to its reader to re-lay.
    constexpr uint32_t FIN_L = get_compile_time_arg_val(15) == 5 ? 3 : get_compile_time_arg_val(15);
    constexpr uint32_t cb_finalized = (COMBINE || FIN_L == 3 || FIN_L == 4) ? cb_rms_root : cb_inv_rms;

    using SumsqPack = ckl::PackTile<ckl::output(
        cb_sumsq_partial,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow)>;
    // x tiles: caller-managed (waited per wave, popped at release), strided {base, row_stride = WS}.
    constexpr auto x_strided_input = ckl::input(
        cb_x_tiles,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    using SquareAccumulateStrided = ckl::
        BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, x_strided_input, ckl::Dst::D0, ckl::DestAccumulation::PerRow>;
    // inv_rms: column-broadcast, addressed from the wave's first row (caller waits cumulatively, pops at release).
    constexpr auto inv_rms_strided_input = ckl::input(
        cb_inv_rms,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto x_gamma_input =
        ckl::input(cb_x_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto scaled_row_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter);
    };
    constexpr auto scaled_shard_output = ckl::output(
        cb_output,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);

    auto collapse = [&](uint32_t n, auto partial_scaler, auto accumulate, auto cb_in_tag, auto cb_out_tag) {
        constexpr uint32_t cb_in = decltype(cb_in_tag)::value;
        constexpr uint32_t cb_out = decltype(cb_out_tag)::value;
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_in,
            cb_reduce_scaler,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::of(n, 1),
            ckl::ReduceInputMemoryLayout::contiguous(),
            accumulate,
            ckl::NoOp{},
            partial_scaler);
    };
    using CbSumsqPartial = std::integral_constant<uint32_t, cb_sumsq_partial>;
    using CbSumsqLast = std::integral_constant<uint32_t, cb_sumsq_last>;
    using CbSumsqAcc = std::integral_constant<uint32_t, cb_sumsq_acc>;
    using CbCollapsed = std::integral_constant<uint32_t, cb_collapsed>;

    // scale_by_gamma of chunk c for the block's R_b rows (x * gamma, gamma row 0 broadcast) -> cb_x_gamma.
    auto scale_by_gamma = [&](uint32_t c, uint32_t rows_in_block) {
        if constexpr (HAS_GAMMA) {
            DRMS_ZONE("cp_gamma");
            // Resident window: the writer pushes it chunk by chunk, chunk c is pages [c*C, +C) (cumulative wait; a
            // no-op after block 0). Streaming (L1 fallback): the writer pushes chunk c per (block, chunk) step as
            // the front C pages, popped right after this chain.
            const uint32_t gamma_off = GAMMA_RESIDENT ? c * C : 0;
            cb_wait_front(cb_gamma_tiles, GAMMA_RESIDENT ? (c + 1) * C : C);
            ckl::eltwise_chain(
                ckl::IterationShape::grid(rows_in_block, C).block_size(SCALE_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_strided_input,
                    ckl::input(
                        cb_gamma_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset)>{ckl::StridedTileRange{c * C, WS}, gamma_off},
                ckl::PackTile<scaled_row_output(cb_x_gamma)>{});
            if constexpr (!GAMMA_RESIDENT) {
                cb_pop_front(cb_gamma_tiles, C);
            }
        }
    };

    const uint32_t num_blocks = (rows_this_core + R - 1) / R;
    const uint32_t full_tiles = masked_core ? slice_valid - 1 : slice_valid;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t remaining = rows_this_core - block_idx * R;
        const uint32_t rows_in_block = remaining < R ? remaining : R;  // R_b
        const uint32_t waves = (rows_in_block + RW - 1) / RW;
        auto wave_n = [&](uint32_t k) { return (rows_in_block - k * RW) < RW ? rows_in_block - k * RW : RW; };

        // ---- reduction side, per wave ----
        for (uint32_t k = 0; k < waves; ++k) {
            const uint32_t row0 = k * RW;
            const uint32_t n = wave_n(k);
            const uint32_t xbase = row0 * WS;
            {
                DRMS_ZONE("cp_xwait");
                cb_wait_front(cb_x_tiles, (row0 + n) * WS);
            }
            DRMS_ZONE("cp_reduce");
            if (full_tiles > 0) {
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(n, full_tiles),
                    SquareAccumulateStrided{ckl::StridedTileRange{xbase, WS}, ckl::StridedTileRange{xbase, WS}},
                    SumsqPack{});
            }
            if constexpr (MASKED_CFG) {
                if (masked_core) {
                    // The partial last tile on its own: its pad columns are dropped by the partial scaler (the
                    // caller's shard is read-only, so they cannot be zeroed in place).
                    const ckl::StridedTileRange last_tile{xbase + slice_valid - 1, WS};
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(n, 1),
                        SquareAccumulateStrided{last_tile, last_tile},
                        ckl::PackTile<ckl::output(
                            cb_sumsq_last,
                            ckl::ReservePolicy::PerOuter,
                            ckl::PushPolicy::PerOuter,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Direct,
                            ckl::DestAccumulation::PerRow)>{});
                    if (full_tiles > 0) {
                        collapse(
                            n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbSumsqAcc{});
                        collapse(
                            n,
                            ckl::ReducePartialScaler::with_partial(),
                            ckl::Accumulate::at(cb_sumsq_acc, 1),
                            CbSumsqLast{},
                            CbCollapsed{});
                    } else {
                        collapse(
                            n,
                            ckl::ReducePartialScaler::with_partial(),
                            ckl::NoAccumulation{},
                            CbSumsqLast{},
                            CbCollapsed{});
                    }
                }
            }
            if (!(MASKED_CFG && masked_core)) {
                collapse(n, ckl::ReducePartialScaler::none(), ckl::NoAccumulation{}, CbSumsqPartial{}, CbCollapsed{});
            }
            if constexpr (COMBINE) {
                if (is_root) {
                    // Device partial of the wave: the G_w member partials of each row (contributor-major slots,
                    // row pitch RW), summed element-wise; no finalize here (the cross-device sum comes first).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_gather,
                        cb_reduce_scaler,
                        cb_dev_partial,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ReduceFp32Mode::Fast,
                        ckl::ReduceAlgorithm::AccumulateViaAdd,
                        ckl::ReduceWithinTile::Skip>(
                        ckl::ReduceInputBlockShape::of(G_W, n),
                        ckl::ReduceInputMemoryLayout::with_row_stride(RW),
                        ckl::NoAccumulation{},
                        ckl::NoOp{});
                }
            }
        }

        // ---- gamma chunk 0, hoisted ahead of the first inv_rms wait (stall shadow of the exchange) ----
        scale_by_gamma(0, rows_in_block);

        // ---- normalize: per chunk, per wave ----
        if constexpr (SHARDED) {
            cb_reserve_back(cb_output, rows_in_block * WS);
        }
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if (c > 0) {
                scale_by_gamma(c, rows_in_block);
            }
            for (uint32_t k = 0; k < waves; ++k) {
                const uint32_t row0 = k * RW;
                const uint32_t n = wave_n(k);
                if (c == 0) {
                    if (is_root) {
                        DRMS_ZONE("cp_fin");
                        finalize_wave<G, cb_finalized>(n, inv_w_bits, eps_bits);
                    }
                    DRMS_ZONE("cp_invwait");
                    cb_wait_front(cb_inv_rms, row0 + n);
                }
                const ckl::StridedTileRange inv_range{row0, 1};
                const ckl::StridedTileRange x_range{row0 * WS + c * C, WS};
                if constexpr (SHARDED) {
                    if constexpr (HAS_GAMMA) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(n, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_strided_input>{0u, inv_range},
                            ckl::PackTile<scaled_shard_output>{x_range});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(n, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_strided_input>{
                                x_range, inv_range},
                            ckl::PackTile<scaled_shard_output>{x_range});
                    }
                } else {
                    if constexpr (HAS_GAMMA) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(n, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_rms_strided_input>{0u, inv_range},
                            ckl::PackTile<scaled_row_output(cb_output)>{});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(n, C).block_size(SCALE_BLOCK),
                            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_rms_strided_input>{
                                x_range, inv_range},
                            ckl::PackTile<scaled_row_output(cb_output)>{});
                    }
                }
            }
            if constexpr (HAS_GAMMA) {
                // Ragged block (R_b < R): this chunk pushed / popped only R_b*C of cb_x_gamma's R*C ring. Advance the
                // ring by the missing (R - R_b)*C pages (nominal push count), so the next chunk's x*gamma window starts
                // at the ring base again: the chains read it by tile index from the read pointer and never wrap
                // (verifier: "Indexed tile read exceeds CB boundary" / garbage output at 8192x3584, R = 3, 2-row
                // cores).
                if (rows_in_block < R) {
                    const uint32_t filler = (R - rows_in_block) * C;
                    cb_reserve_back(cb_x_gamma, filler);
                    cb_push_back(cb_x_gamma, filler);
                    cb_wait_front(cb_x_gamma, filler);
                    cb_pop_front(cb_x_gamma, filler);
                }
            }
        }
        if constexpr (SHARDED) {
            cb_push_back(cb_output, rows_in_block * WS);
        }

        // ---- release_block ----
        cb_pop_front(cb_x_tiles, rows_in_block * WS);
        cb_pop_front(cb_inv_rms, rows_in_block);
    }

    // reduce() waits the scaler but never pops it; the caller owns that page (or pair).
    cb_pop_front(cb_reduce_scaler, MASKED_CFG ? 2 : 1);
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        if (num_blocks > 0) {
            cb_pop_front(cb_gamma_tiles, Wp);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr bool has_gamma = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t x_row_stride = get_compile_time_arg_val(4);  // WS
    constexpr bool sharded = get_compile_time_arg_val(5) != 0;
    constexpr bool combine = get_compile_time_arg_val(6) != 0;
    constexpr bool masked_cfg = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t group_cores = get_compile_time_arg_val(8);  // G_w
    constexpr uint32_t scale_block_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t wave_rows = get_compile_time_arg_val(10);      // Rw
    constexpr uint32_t group_devices = get_compile_time_arg_val(12);  // G
    constexpr bool gamma_resident = get_compile_time_arg_val(13) != 0;

    const uint32_t rows_this_core = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);
    const uint32_t slice_valid = get_arg_val<uint32_t>(3);
    const bool is_root = get_arg_val<uint32_t>(4) != 0;
    const bool masked_core = get_arg_val<uint32_t>(5) != 0;

    compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial);

    run<has_gamma,
        block_rows,
        w_chunk_tiles,
        num_w_chunks,
        x_row_stride,
        sharded,
        combine,
        masked_cfg,
        group_cores,
        scale_block_tiles,
        wave_rows,
        group_devices,
        gamma_resident>(rows_this_core, inv_w_bits, eps_bits, slice_valid, is_root, masked_core);
}
