// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm compute (TRISC0/1/2). Block operations (op_design.md -> Block Operation Realization):
//
// RMS:
//   local_stats_block    sum_squares_block (x*x DEST-accumulated per tile-row, FPU) -> cb_sumsq
//                        reduce<SUM, ROW, ReduceTile, Accurate>(crt x 1) -> cb_partial (raw sum x^2, column 0, Float32)
//   combine_stats_block  reduce<SUM, ROW, ReduceTile, Accurate>(crt x gather_tiles) over the P = G*k partition
//                        columns + finalize rsqrt(sum / W + eps) -> cb_inv (fp32 SFPU sums: prompt precision rule)
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
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
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

// ---- transpose_gathered: row-form gather tiles (row j = partition j) -> column form (column j) ----
// Raw compute API (documented helper bypass): no kernel_lib helper transposes a tile in DEST. copy_tile from the
// UnpackToDestFp32-tagged cb_gather_rows and transpose_dest keep every fp32 bit (an unpacker / FPU transpose would
// round the partial sums to tf32 — the LN mean records must stay exact). Base op mechanism (transpose_dest).
ALWI void transpose_gathered(uint32_t n) {
    cb_wait_front(cb_gather_rows, n);
    cb_reserve_back(cb_gathered, n);
    reconfig_data_format_srca(cb_gather_rows);
    pack_reconfig_data_format(cb_gathered);
    for (uint32_t t = 0; t < n; ++t) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_gather_rows);
        copy_tile(cb_gather_rows, t, 0);
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gathered);
        tile_regs_release();
    }
    cb_push_back(cb_gathered, n);
    cb_pop_front(cb_gather_rows, n);
}

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

// Exact fp32 sums of the RMS partials need fp32 DEST (UnpackToDestFp32-tagged Float32 CBs, host side).
constexpr ReduceFp32Mode partial_sum_fp32_mode = DST_ACCUM_MODE ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

// A sharded input's W tail is never zeroed in the caller's shard: on the member that owns the tail, every sum over x
// multiplies its block by a row-broadcast mask block (cb_mask: cw row-0 masks, ones but the tail tile's valid
// columns; DestReuse, like gamma) before a plain reduce. (AccumulateViaAdd's masked fold of a lone tail tile — cw == 1
// — accumulates onto an unseeded DEST when crt > 1, so the mask is applied in the chain instead.)
constexpr bool masked_tail_path = x_sharded && (w_valid_last < TILE);
constexpr auto mask_in = input(
    cb_mask,
    WaitPolicy::None,
    PopPolicy::None,
    InputTileMapping::Row,
    DataFormatReconfig::Enabled,
    TileAddressing::Direct);
using TimesMask = DestReuseRowBcast<mask_in, EltwiseBinaryType::ELWMUL, BinaryFpuOp::Mul>;

// Row sums of the (already masked) cb_ln_sq block -> cb_dst (column 0), consuming it.
template <uint32_t cb_dst, typename Post>
ALWI void masked_block_row_sum(uint32_t crt, uint32_t cw, Post post) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_ln_sq,
        cb_scaler,
        cb_dst,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(crt, cw), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post);
}

ALWI void rms_block(const WorkerId& w, FinalizeInvRms finalize, bool tail) {
    const uint32_t crt = w.crt, cw = w.cw;
    cb_wait_front(cb_x, crt * cw);
    // local_stats_block
    if constexpr (masked_tail_path) {
        if (tail) {
            // masked x^2 tiles of the block (padding lanes of the tail tile x 0), then the row sums
            cb_wait_front(cb_mask, cw);
            eltwise_chain(
                IterationShape::grid(crt, cw),
                CopyTile<input(cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0>{},
                Square<Dst::D0>{},
                TimesMask{},
                PackTile<output(cb_ln_sq)>{});
            masked_block_row_sum<cb_partial>(crt, cw, NoOp{});
        }
    }
    if (!masked_tail_path || !tail) {
        sum_squares_block(crt, cw);
        // Partial sums stay fp32 when summed (prompt precision contract): with fp32 DEST, cb_sumsq is a Float32
        // UnpackToDestFp32 CB and the Accurate (SFPU) reduce copies it into DEST exactly; the FPU path would truncate
        // every per-tile-row sum to tf32 before the collapse.
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sumsq,
            cb_scaler,
            cb_partial,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            partial_sum_fp32_mode,
            ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(crt, 1));
    }
    // combine_stats_block (waits for the reader's gather tiles)
    transpose_gathered(crt * gather_tiles);
    // The P = G*k gathered partials are summed in fp32 (SFPU, exact copies from the UnpackToDestFp32 cb_gathered):
    // AccumulateViaAdd / the FPU reduce would read them through srcA as tf32.
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_scaler,
        cb_inv,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        partial_sum_fp32_mode,
        ReduceAlgorithm::ReduceTile>(
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

// The x block the LN paths read: fp32 DEST -> a Float32 block (UnpackToDestFp32: exact in DEST); 16-bit DEST -> cb_x
// itself (nothing is UnpackToDestFp32-tagged, every DEST value is 16-bit anyway).
constexpr bool ln_x_copy = !x_is_f32 && DST_ACCUM_MODE;
constexpr uint32_t cb_xs = ln_x_copy ? cb_x32 : cb_x;
// Exact (SFPU) fp32 sums need fp32 DEST; under 16-bit DEST the FPU path sums (the caller chose 16-bit accumulation).
constexpr ReduceFp32Mode ln_exact_mode = DST_ACCUM_MODE ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

// Row sums of the resident x block with the W-tail tile masked (sharded input: the caller's shard keeps its padding).
template <uint32_t cb_dst>
ALWI void masked_row_sum(uint32_t crt, uint32_t cw, MulScalarPost post) {
    eltwise_chain(
        IterationShape::grid(crt, cw),
        CopyTile<input(cb_xs, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0>{},
        TimesMask{},
        PackTile<output(cb_ln_sq)>{});
    masked_block_row_sum<cb_dst>(crt, cw, post);
}

// LN combine over the gather tiles of a row: one gather tile -> AccumulateViaAdd's in-tile collapse (Phase 0 path);
// several -> the exact ReduceTile path the RMS combine uses (AccumulateViaAdd's cross-tile add_tiles would read the
// UnpackToDestFp32-tagged cb_ln_tmp through srcA).
constexpr ReduceAlgorithm ln_comb_alg =
    gather_tiles > 1 ? ReduceAlgorithm::ReduceTile : ReduceAlgorithm::AccumulateViaAdd;
constexpr ReduceFp32Mode ln_comb_mode = gather_tiles > 1 ? ln_exact_mode : ReduceFp32Mode::Fast;

ALWI void ln_block(const WorkerId& w, FinalizeInvRms finalize, uint32_t inv_n_bits, uint32_t inv_w_bits, bool tail) {
    const uint32_t crt = w.crt, cw = w.cw;
    cb_wait_front(cb_x, crt * cw);
    if constexpr (ln_x_copy) {
        eltwise_chain(
            IterationShape::grid(crt, cw),
            CopyTile<input(cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block)>{},
            PackTile<output(cb_x32)>{});
        cb_pop_front(cb_x, crt * cw);
        cb_wait_front(cb_x32, crt * cw);
    }
    {
        AGRN_ZONE("c_local_start");
    }
    // ---- local_stats_block ----
    // m (stat 0), then a second copy of m for the shift broadcast
    if constexpr (masked_tail_path) {
        if (tail) {
            // sharded input, tail member: S1 over the valid columns only (the shard is not zeroed)
            cb_wait_front(cb_mask, cw);
            masked_row_sum<cb_partial>(crt, cw, MulScalarPost{inv_n_bits});
            masked_row_sum<cb_ln_m>(crt, cw, MulScalarPost{inv_n_bits});
        }
    }
    if (!masked_tail_path || !tail) {
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_xs,
            cb_scaler,
            cb_partial,
            ReduceInputPolicy::NoWaitNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ln_exact_mode,
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
            ln_exact_mode,
            ReduceAlgorithm::ReduceTile>(
            ReduceInputBlockShape::of(crt, cw),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            MulScalarPost{inv_n_bits});
    }
    // K = tf32(m) in every column
    unary_bcast<BroadcastDim::Col, input(cb_ln_m), output(cb_ln_shift)>(IterationShape::tiles(crt));
    cb_wait_front(cb_ln_shift, crt);
    // T = sum (x - K)^2 (padding columns masked in the reduce; sharded tail: by the mask block in the chain)
    constexpr auto xs_blk = input(cb_xs, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    constexpr auto shift_col = input(cb_ln_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    if constexpr (masked_tail_path) {
        if (tail) {
            eltwise_chain(
                IterationShape::grid(crt, cw),
                CopyTile<xs_blk, Dst::D0>{},
                CopyTile<shift_col, Dst::D1>{},
                SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                Square<Dst::D0>{},
                TimesMask{},
                PackTile<output(cb_ln_sq)>{});
            masked_block_row_sum<cb_partial>(crt, cw, NoOp{});
        }
    }
    if (!masked_tail_path || !tail) {
        eltwise_chain(
            IterationShape::grid(crt, cw),
            CopyTile<xs_blk, Dst::D0>{},
            CopyTile<shift_col, Dst::D1>{},
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
    }
    // K (stat 2): tf32-exact, so the copy is exact
    eltwise_chain(
        IterationShape::tiles(crt),
        CopyTile<input(cb_ln_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block)>{},
        PackTile<output(cb_partial)>{});
    // cb_ln_shift (K in every column, exact) stays: it is the centering shift of the normalize

    {
        AGRN_ZONE("c_combine_start");
    }
    // ---- combine_stats_block ----
    // gathered per row: [m, T, K] x gather_tiles (partition j -> column j % 32 of gather tile j / 32), all
    // UnpackToDestFp32 under fp32 DEST (exact copies). Padding columns j >= P carry n_j = 0 and contribute 0.
    {
        AGRN_ZONE("c_gather_in");
    }
    transpose_gathered(crt * nstats * gather_tiles);
    {
        AGRN_ZONE("c_transposed");
    }
    cb_wait_front(cb_gathered, crt * nstats * gather_tiles);
    cb_wait_front(cb_stat_consts, 1);
    constexpr auto g_m = input(
        cb_gathered,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto g_n =
        input(cb_stat_consts, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);  // n_j of gather tile c
    eltwise_chain(
        IterationShape::grid(crt, gather_tiles),
        CopyTile<g_m, Dst::D0>{StridedTileRange{0 * gather_tiles, nstats * gather_tiles}},
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
        ln_comb_mode,
        ln_comb_alg>(
        ReduceInputBlockShape::of(crt, gather_tiles),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        MulScalarPost{inv_w_bits});
    {
        AGRN_ZONE("c_mean_done");
    }
    // d = mean - K (exact in fp32 SFPU), then broadcast: x - mean = (x - K) - d with both K (tf32-exact) and the small
    // d (its tf32 truncation is ~1e-3 of d, i.e. of the partition-mean scatter, not of the mean) safe to broadcast
    // on the FPU — no exact fp32 broadcast of the mean is needed (op_design.md Key Risks: LN precision).
    constexpr auto shift_in = input(cb_ln_shift, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    cb_wait_front(cb_mean_col, crt);
    eltwise_chain(
        IterationShape::grid(crt, 1),
        CopyTile<input(cb_mean_col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col), Dst::D0>{},
        CopyTile<shift_in, Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<output(cb_ln_m)>{});
    cb_pop_front(cb_mean_col, crt);
    unary_bcast<BroadcastDim::Col, input(cb_ln_m), output(cb_mean_b)>(IterationShape::tiles(crt));
    cb_wait_front(cb_mean_b, crt);
    {
        AGRN_ZONE("c_meanb_in");
    }
    constexpr auto mean_b_in = input(cb_mean_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    eltwise_chain(
        IterationShape::grid(crt, gather_tiles),
        CopyTile<g_m, Dst::D0>{StridedTileRange{0 * gather_tiles, nstats * gather_tiles}},
        CopyTile<g_m, Dst::D1>{StridedTileRange{2 * gather_tiles, nstats * gather_tiles}},  // K
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Square<Dst::D0>{},
        CopyTile<g_n, Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},                                             // n (m - K)^2
        CopyTile<g_m, Dst::D1>{StridedTileRange{1 * gather_tiles, nstats * gather_tiles}},  // T
        SubBinary<Dst::D1, Dst::D0, Dst::D1>{},  // T - n (m - K)^2 = sum (x - m)^2
        CopyTile<g_m, Dst::D0>{StridedTileRange{0 * gather_tiles, nstats * gather_tiles}},
        CopyTile<shift_in, Dst::D2>{},
        SubBinary<Dst::D0, Dst::D2, Dst::D0>{},  // m - K
        CopyTile<mean_b_in, Dst::D2>{},
        SubBinary<Dst::D0, Dst::D2, Dst::D0>{},  // m - mean = (m - K) - d
        Square<Dst::D0>{},
        CopyTile<g_n, Dst::D2>{},
        MulBinary<Dst::D0, Dst::D2, Dst::D0>{},  // n (m - mean)^2
        AddBinary<Dst::D1, Dst::D0, Dst::D1>{},
        PackTile<output(cb_ln_tmp), Dst::D1>{});
    {
        AGRN_ZONE("c_m2_done");
    }
    cb_pop_front(cb_gathered, crt * nstats * gather_tiles);
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_ln_tmp,
        cb_scaler,
        cb_inv,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ln_comb_mode,
        ln_comb_alg>(
        ReduceInputBlockShape::of(crt, gather_tiles),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        finalize);
    unary_bcast<BroadcastDim::Col, input(cb_inv), output(cb_rstd_b)>(IterationShape::tiles(crt));
    cb_wait_front(cb_rstd_b, crt);

    {
        AGRN_ZONE("c_rstd_done");
    }
    // ---- normalize_block ----
    constexpr auto xs_in = input(cb_xs, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    constexpr auto rstd_in = input(cb_rstd_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using LoadX = CopyTile<xs_in, Dst::D0>;
    using LoadShift = CopyTile<shift_in, Dst::D1>;
    using LoadMean = CopyTile<mean_b_in, Dst::D1>;  // d = mean - K
    using LoadRstd = CopyTile<rstd_in, Dst::D1>;
    using Center = SubBinary<Dst::D0, Dst::D1, Dst::D0>;
    using Scale = MulBinary<Dst::D0, Dst::D1, Dst::D0>;
    if constexpr (has_gamma && has_beta) {
        cb_wait_front(cb_gamma, cw);
        cb_wait_front(cb_beta, cw);
        eltwise_chain(
            IterationShape::grid(crt, cw),
            LoadX{},
            LoadShift{},
            Center{},
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
            LoadShift{},
            Center{},
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
            LoadShift{},
            Center{},
            LoadMean{},
            Center{},
            LoadRstd{},
            Scale{},
            PackTile<output(cb_out_block)>{});
    }
    {
        AGRN_ZONE("c_norm_done");
    }
    cb_pop_front(cb_rstd_b, crt);
    cb_pop_front(cb_mean_b, crt);
    cb_pop_front(cb_ln_shift, crt);
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
        rms_block(w, finalize, member_owns_tail(w));
    }
}
