// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm compute (TRISC0/1/2). Block operations (op_design.md -> Block Operation Realization):
//
// RMS:
//   local_stats_block    sum_squares_block (x*x DEST-accumulated per tile-row, FPU) -> cb_sumsq
//                        reduce<SUM, ROW, ReduceTile>(crt x 1) -> cb_partial (raw sum x^2, column 0, Float32)
//   combine_stats_block  reduce<SUM, ROW, AccumulateViaAdd, Collapse>(crt x gather_tiles) over the P = G*k partition
//                        columns + finalize rsqrt(sum / W + eps) -> cb_inv
//   normalize_block      eltwise_chain BinaryFpu(x * inv bcast Col) [-> MulGammaRowBcast] -> cb_out_block
// LN (shifted sums, one stats round, exact in fp32 — op_design.md Key Risks):
//   [copy x -> cb_x32]   non-Float32 x only: an exact Float32 copy (UnpackToDestFp32 consumers need Float32)
//   local_stats_block    m  = reduce<SUM, ROW, Accurate>(x) * (1/n)  -> cb_partial (stat 0) and cb_ln_m
//                        K  = UnaryBcast<Col>(m) (= tf32(m), exact in every column) -> cb_ln_shift
//                        T  = reduce<AccumulateViaAdd>( (x - K)^2 [W tail masked] ) -> cb_partial (stat 1)
//                        K column -> cb_partial (stat 2)
//   combine_stats_block  mean = sum_j n_j m_j / W                     -> cb_mean_col (reader broadcasts it exactly)
//                        M2   = sum_j [T_j - n_j (m_j - K_j)^2 + n_j (m_j - mean)^2] -> rstd -> cb_inv -> cb_rstd_b
//   normalize_block      (x - mean) * rstd [* gamma] [+ beta], x / mean exact fp32 in DEST (SFPU) -> cb_out_block
//
// Raw API / raw LLK (documented exceptions, ported from the base op rms_norm_compute.cpp):
//  * sum_squares_block: eltwise_chain's DestAccumulation::PerRow packs one tile-row per acquire; the base op's raw
//    batched loop (DEST_AUTO_LIMIT tile-rows per acquire) is ported unchanged.
//  * MulGammaRowBcast / AddBetaRowBcast: DestReuseBinary has no BroadcastDim; the LLK supports a ROW-broadcast CB
//    operand on the dest-reuse op, which lets cb_gamma / cb_beta keep only row 0 of faces 0/1.
//  * FinalizeInvRms: column-0-scoped SFPU body (x * 1/W + eps -> rsqrt) as the reduce() post_reduce_op.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif
#include "agrn_worker_args.hpp"

using namespace compute_kernel_lib;
using namespace agrn;

// rsqrt(sum * (1/W) + eps) on column 0 of DST[dst]: VectorMode::C visits faces 0/2, the body the even-parity
// vectors (column 0 lives only there). Base op FinalizeInvRms.
#ifdef TRISC_MATH
sfpi_inline void finalize_col0_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int rg = 0; rg < 4; rg++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat y =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] * inv_w + eps);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;
    }
}
#endif

struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, inv_w_bits, eps_bits)));
    }
};

struct MulScalarPost {
    uint32_t bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, bits);
    }
};

// DEST[d] (*|+)= row-broadcast(cb tile): a DestReuseBinary<op, In, DEST_TO_SRCA> whose CB operand (srcB) is
// ROW-broadcast (base op MulGammaRowBcast). Only row 0 of faces 0/1 of the CB page is read.
template <InputSpec In, EltwiseBinaryType ElwOp, BinaryFpuOp ChainOp>
struct DestReuseRowBcast : DestReuseBinary<ChainOp, In, DestReuseType::DEST_TO_SRCA> {
    using Base = DestReuseBinary<ChainOp, In, DestReuseType::DEST_TO_SRCA>;
    static constexpr uint32_t GCb = In.cb_id;
    constexpr DestReuseRowBcast() noexcept = default;
    constexpr explicit DestReuseRowBcast(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        state_configure(GCb, __builtin_LINE());
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, GCb)));
        MATH((llk_math_eltwise_binary_init<
              ElwOp,
              BroadcastType::ROW,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, 0)));
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx =
            compute_kernel_lib::tile_base_value<Base::Addressing>(this->tile_base) +
            compute_kernel_lib::detail::input_idx<Base::Mapping, Base::Wait, Base::Pop, Base::Addressing>(
                i_flat, ht, wt, this->row_stride);
        const uint32_t dst = to_u32(Base::DstSlot) + slot_offset;
        UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, in_idx)));
        MATH((llk_math_eltwise_binary<
              ElwOp,
              BroadcastType::ROW,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::DEST_TO_SRCA>(GCb, GCb, dst, true)));
    }
};

constexpr auto gamma_in = input(
    cb_gamma,
    WaitPolicy::None,
    PopPolicy::None,
    InputTileMapping::Row,
    DataFormatReconfig::Enabled,
    TileAddressing::Direct);
constexpr auto beta_in = input(
    cb_beta,
    WaitPolicy::None,
    PopPolicy::None,
    InputTileMapping::Row,
    DataFormatReconfig::Enabled,
    TileAddressing::Direct);
using TimesGamma = DestReuseRowBcast<gamma_in, EltwiseBinaryType::ELWMUL, BinaryFpuOp::Mul>;
using PlusBeta = DestReuseRowBcast<beta_in, EltwiseBinaryType::ELWADD, BinaryFpuOp::Add>;

// ---- RMS pass 1: per tile-row sum_j x[r][j]^2 accumulated in DEST (base op sum_squares_block) ----
ALWI void sum_squares_block(uint32_t rows, uint32_t cw) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_sumsq);
    mul_init(cb_x, cb_x, true);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_reserve_back(cb_sumsq, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = (r0 + i) * cw;
            for (uint32_t j = 0; j < cw; ++j) {
                mul_tiles(cb_x, cb_x, base + j, base + j, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_sumsq);
        }
        tile_regs_release();
        cb_push_back(cb_sumsq, n);
    }
}

ALWI void rms_block(const WorkerId& w, FinalizeInvRms finalize) {
    const uint32_t crt = w.crt, cw = w.cw;
    cb_wait_front(cb_x, crt * cw);
    // local_stats_block
    sum_squares_block(crt, cw);
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq,
        cb_scaler,
        cb_partial,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(crt, 1));
    // combine_stats_block (waits for the reader's gather tiles)
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_scaler,
        cb_inv,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(
        ReduceInputBlockShape::of(crt, gather_tiles),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        finalize);
    // normalize_block
    cb_wait_front(cb_inv, crt);
    constexpr auto x_in = input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct);
    constexpr auto inv_in = input(cb_inv, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XTimesInv = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_in>;
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, cw);
        eltwise_chain(IterationShape::grid(crt, cw), XTimesInv{}, TimesGamma{}, PackTile<output(cb_out_block)>{});
        cb_pop_front(cb_gamma, cw);
    } else {
        eltwise_chain(IterationShape::grid(crt, cw), XTimesInv{}, PackTile<output(cb_out_block)>{});
    }
    cb_pop_front(cb_inv, crt);
    cb_pop_front(cb_x, crt * cw);
}

// The Float32 x block the LN SFPU paths read (UnpackToDestFp32: exact in DEST).
constexpr uint32_t cb_xs = x_is_f32 ? cb_x : cb_x32;

ALWI void ln_block(const WorkerId& w, FinalizeInvRms finalize, uint32_t inv_n_bits, uint32_t inv_w_bits, bool tail) {
    const uint32_t crt = w.crt, cw = w.cw;
    cb_wait_front(cb_x, crt * cw);
    if constexpr (!x_is_f32) {
        eltwise_chain(
            IterationShape::grid(crt, cw),
            CopyTile<input(cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block)>{},
            PackTile<output(cb_x32)>{});
        cb_pop_front(cb_x, crt * cw);
        cb_wait_front(cb_x32, crt * cw);
    }
    // ---- local_stats_block ----
    // m (stat 0), then a second copy of m for the shift broadcast
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_xs,
        cb_scaler,
        cb_partial,
        ReduceInputPolicy::NoWaitNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(crt, cw),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        MulScalarPost{inv_n_bits});
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_xs,
        cb_scaler,
        cb_ln_m,
        ReduceInputPolicy::NoWaitNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(crt, cw),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        MulScalarPost{inv_n_bits});
    // K = tf32(m) in every column
    unary_bcast<BroadcastDim::Col, input(cb_ln_m), output(cb_ln_shift)>(IterationShape::tiles(crt));
    cb_wait_front(cb_ln_shift, crt);
    // T = sum (x - K)^2 (padding columns masked in the reduce)
    eltwise_chain(
        IterationShape::grid(crt, cw),
        CopyTile<input(cb_xs, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0>{},
        CopyTile<input(cb_ln_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col), Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Square<Dst::D0>{},
        PackTile<output(cb_ln_sq)>{});
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_ln_sq,
        cb_mask,
        cb_partial,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd>(
        ReduceInputBlockShape::of(crt, cw),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        NoOp{},
        tail ? ReducePartialScaler::only_partial() : ReducePartialScaler::none());
    // K (stat 2): tf32-exact, so the copy is exact
    eltwise_chain(
        IterationShape::tiles(crt),
        CopyTile<input(cb_ln_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block)>{},
        PackTile<output(cb_partial)>{});
    cb_pop_front(cb_ln_shift, crt);

    // ---- combine_stats_block ----
    // gathered per row: [m, T, K] (gather_tiles == 1 for LN), all UnpackToDestFp32 (exact copies)
    cb_wait_front(cb_gathered, crt * nstats);
    cb_wait_front(cb_stat_consts, 1);
    constexpr auto g_m = input(
        cb_gathered,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Col,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto g_n = input(cb_stat_consts, WaitPolicy::None, PopPolicy::None, InputTileMapping::Scalar);
    eltwise_chain(
        IterationShape::grid(crt, 1),
        CopyTile<g_m, Dst::D0>{StridedTileRange{0, nstats}},
        CopyTile<g_n, Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<output(cb_ln_tmp)>{});
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_ln_tmp,
        cb_scaler,
        cb_mean_col,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(
        ReduceInputBlockShape::of(crt, 1),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        MulScalarPost{inv_w_bits});
    // the reader broadcasts the exact mean -> cb_mean_b
    cb_wait_front(cb_mean_b, crt);
    constexpr auto mean_b_in = input(cb_mean_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    eltwise_chain(
        IterationShape::grid(crt, 1),
        CopyTile<g_m, Dst::D0>{StridedTileRange{0, nstats}},
        CopyTile<g_m, Dst::D1>{StridedTileRange{2, nstats}},  // K
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Square<Dst::D0>{},
        CopyTile<g_n, Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},               // n (m - K)^2
        CopyTile<g_m, Dst::D1>{StridedTileRange{1, nstats}},  // T
        SubBinary<Dst::D1, Dst::D0, Dst::D1>{},               // T - n (m - K)^2 = sum (x - m)^2
        CopyTile<g_m, Dst::D0>{StridedTileRange{0, nstats}},
        CopyTile<mean_b_in, Dst::D2>{},
        SubBinary<Dst::D0, Dst::D2, Dst::D0>{},
        Square<Dst::D0>{},
        CopyTile<g_n, Dst::D2>{},
        MulBinary<Dst::D0, Dst::D2, Dst::D0>{},  // n (m - mean)^2
        AddBinary<Dst::D1, Dst::D0, Dst::D1>{},
        PackTile<output(cb_ln_tmp), Dst::D1>{});
    cb_pop_front(cb_gathered, crt * nstats);
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_ln_tmp,
        cb_scaler,
        cb_inv,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(
        ReduceInputBlockShape::of(crt, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, finalize);
    unary_bcast<BroadcastDim::Col, input(cb_inv), output(cb_rstd_b)>(IterationShape::tiles(crt));
    cb_wait_front(cb_rstd_b, crt);

    // ---- normalize_block ----
    constexpr auto xs_in = input(cb_xs, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    constexpr auto rstd_in = input(cb_rstd_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using LoadX = CopyTile<xs_in, Dst::D0>;
    using LoadMean = CopyTile<mean_b_in, Dst::D1>;
    using LoadRstd = CopyTile<rstd_in, Dst::D1>;
    using Center = SubBinary<Dst::D0, Dst::D1, Dst::D0>;
    using Scale = MulBinary<Dst::D0, Dst::D1, Dst::D0>;
    if constexpr (has_gamma && has_beta) {
        cb_wait_front(cb_gamma, cw);
        cb_wait_front(cb_beta, cw);
        eltwise_chain(
            IterationShape::grid(crt, cw),
            LoadX{},
            LoadMean{},
            Center{},
            LoadRstd{},
            Scale{},
            TimesGamma{},
            PlusBeta{},
            PackTile<output(cb_out_block)>{});
        cb_pop_front(cb_gamma, cw);
        cb_pop_front(cb_beta, cw);
    } else if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, cw);
        eltwise_chain(
            IterationShape::grid(crt, cw),
            LoadX{},
            LoadMean{},
            Center{},
            LoadRstd{},
            Scale{},
            TimesGamma{},
            PackTile<output(cb_out_block)>{});
        cb_pop_front(cb_gamma, cw);
    } else {
        eltwise_chain(
            IterationShape::grid(crt, cw),
            LoadX{},
            LoadMean{},
            Center{},
            LoadRstd{},
            Scale{},
            PackTile<output(cb_out_block)>{});
    }
    cb_pop_front(cb_rstd_b, crt);
    cb_pop_front(cb_mean_b, crt);
    cb_pop_front(cb_xs, crt * cw);
}

void kernel_main() {
    const WorkerId w = my_worker_id();
    if (!w.active) {
        return;
    }
    const FinalizeInvRms finalize{
        get_common_arg_val<uint32_t>(CRT_INV_W_BITS), get_common_arg_val<uint32_t>(CRT_EPS_BITS)};
    if constexpr (is_ln) {
        compute_kernel_hw_startup(cb_xs, cb_scaler, cb_partial);
        ln_block(
            w,
            finalize,
            get_common_arg_val<uint32_t>(CRT_INV_N + w.m),
            get_common_arg_val<uint32_t>(CRT_INV_W_BITS),
            member_owns_tail(w));
    } else {
        compute_kernel_hw_startup(cb_x, cb_x, cb_sumsq);
        rms_block(w, finalize);
    }
}
