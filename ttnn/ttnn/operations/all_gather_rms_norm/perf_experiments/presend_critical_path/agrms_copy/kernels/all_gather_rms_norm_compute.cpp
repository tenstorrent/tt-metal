// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm compute (TRISC0/1/2). The local half, ported from the base op rms_norm
// (ttnn/ttnn/operations/base_ops/rms_norm/kernels/rms_norm_compute.cpp). Block operations per block of
// block_row_tiles local tile-rows (op_design.md -> Block Operation Realization):
//   sum_squares_block       raw (ported): x*x DEST-accumulated (FPU) per tile-row over the member's landing tiles,
//                           DEST_AUTO_LIMIT tile-rows per acquire -> cb_sumsq_tiles
//   collapse_sumsq_block    reduce<SUM, ROW, ReduceTile> (1.0 scaler cb_ones) -> cb_inv_rms (k == 1, + finalize)
//                                                                            -> cb_partial_sumsq (k > 1)
//   combine_partials_block  k > 1 (symmetric): reduce<SUM, ROW, AccumulateViaAdd, Skip>(rows x k) + finalize
//                           -> cb_inv_rms
//   normalize_block         eltwise_chain BinaryFpu(x * inv_rms, bcast Col) -> MulGammaRowBcast -> cb_out_tiles
//                           regime A: x = the resident landing tiles (Strided); regime N: x = the reader's composed
//                           dense tiles (cb_x_aligned, streamed per tile).
//
// Raw API / raw LLK (documented exceptions, inherited from the base op and measured there):
//  * MulGammaRowBcast — DestReuseBinary has no BroadcastDim (mul_reuse_dest_* hard-code BroadcastType::NONE); the LLK
//    supports a ROW-broadcast CB operand on the dest-reuse multiply, which lets cb_gamma hold only row 0 of faces 0/1.
//  * FinalizeInvRms — column-0-scoped SFPU body (x * 1/W + eps -> rsqrt over faces 0/2 even vectors) run through the
//    reduce() post_reduce_op hook; the stock SFPU entry points scope only by face, never by column parity.
//  * sum_squares_block — eltwise_chain's DestAccumulation::PerRow packs one tile-row per acquire; the base op's
//    batched raw path (DEST_AUTO_LIMIT rows per acquire) is ported as is.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "api/compute/reduce.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/matmul.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif
#include "all_gather_rms_norm_common.hpp"

using namespace compute_kernel_lib;
using namespace agrms_cb;

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);  // k
constexpr uint32_t w_pitch = get_compile_time_arg_val(1);        // landing pages per local row
constexpr uint32_t block_row_tiles = get_compile_time_arg_val(2);
constexpr bool regime_n = get_compile_time_arg_val(3) != 0;
constexpr uint32_t out_unit = get_compile_time_arg_val(4);
constexpr uint32_t out_w_tiles = get_compile_time_arg_val(5);
constexpr bool combine_owner = get_compile_time_arg_val(6) != 0;  // k > K_SYM_MAX: per-row owner combine
constexpr uint32_t row_gather = get_compile_time_arg_val(7);      // owner: gather tiles per owned row, ceil(k / 32)
constexpr bool own_in_shard = get_compile_time_arg_val(8) != 0;   // native HEIGHT: own slice read from cb_x_in
constexpr uint32_t Wt_lp = get_compile_time_arg_val(9);           // tiles per slice (cb_x_in row pitch)
constexpr bool stage = get_compile_time_arg_val(10) != 0;         // regime N bfp8: bf16 staging row for the compose
// OVERLAP (Refinement 4): x * gamma is computed into cb_x_gamma while the k partials are exchanged (the combine wait),
// so the normalize after the combine is one multiply (x*gamma) * inv_rms instead of two. Regime A, k > 1, gamma.
constexpr bool prescale = get_compile_time_arg_val(11) != 0;
// Local work (Refinement 5): symmetric combine on ROW-form partials (the owner path's collapse_rows_to_row0 /
// combine_rows_block, every member combining every row): the exchange ships 2 face rows per message, not 2 faces.
constexpr bool row_partials = get_compile_time_arg_val(12) != 0;

// rsqrt(sum * (1/W) + eps) on column 0 of DST[dst] — the reduce() post_reduce_op (base op FinalizeInvRms).
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
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1,3,..,15 - never column 0)
    }
}
#endif

struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        rsqrt_tile_init();  // sqrt-body constants
        MATH((_llk_math_eltwise_unary_sfpu_params_(finalize_col0_body, dst, VectorMode::C, inv_w_bits, eps_bits)));
    }
};

// ---- sum_squares_block: per tile-row, sum_j x[r][j]^2 accumulated in DEST, one tile per tile-row ----
// The block's tile-row i starts at page i * w_pitch of the cb_x_landing front (earlier blocks are popped).
// own_in_shard: landing columns [own_c0, own_c0 + Wt_lp) are read from the resident input shard instead (cb_x_in,
// waited once for every row; local row i0 + i at page (i0 + i) * Wt_lp).
ALWI void sum_squares_block(uint32_t i0, uint32_t rows, uint32_t full_w, uint32_t own_c0) {
    cb_wait_front(cb_x_landing, rows * w_pitch);
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    reconfig_data_format(cb_x_landing, cb_x_landing);
    pack_reconfig_data_format(cb_sumsq_tiles);
    mul_init(cb_x_landing, cb_x_landing, true);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_reserve_back(cb_sumsq_tiles, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = (r0 + i) * w_pitch;
            for (uint32_t j = 0; j < full_w; ++j) {
                if (own_in_shard && j - own_c0 < Wt_lp) {
                    const uint32_t xi = (i0 + r0 + i) * Wt_lp + (j - own_c0);
                    mul_tiles(cb_x_in, cb_x_in, xi, xi, i);
                } else {
                    mul_tiles(cb_x_landing, cb_x_landing, base + j, base + j, i);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_sumsq_tiles);
        }
        tile_regs_release();
        cb_push_back(cb_sumsq_tiles, n);
    }
}

// ---- collapse_sumsq_block: within-tile row collapse on the FPU (column 0 = row sum) [+ finalize] ----
template <uint32_t out_cb, typename PostOp = NoOp>
ALWI void collapse_sumsq_block(uint32_t rows, PostOp post_op) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq_tiles,
        cb_ones,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op);
}

// ---- combine_partials_block: sum the k gathered column-0 partials per tile-row + finalize ----
ALWI void combine_partials_block(uint32_t rows, FinalizeInvRms finalize) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered_sumsq,
        cb_ones,  // scaler slot: never read by AccumulateViaAdd + Skip
        cb_inv_rms,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Skip>(
        ReduceInputBlockShape::of(rows, w_split_cores),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        finalize);
}

// ---- Owner combine (k > K_SYM_MAX), ported from the base op (Perf 2 row_partial_mcast). Raw compute API (documented
// helper bypass, base op): reduce() has no row-form output and no "REDUCE_COL, then DEST transpose, then finalize".
// collapse_rows_to_row0: per tile-row one FPU matmul ones-row x sumsq^T (unpacker transpose) -> row 0 = the 32 row
// sums (faces 0/1 row 0), which the writer ships as two face rows into row (m % 32) of the owner's gather tile m / 32.
ALWI void collapse_rows_to_row0(uint32_t rows) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    cb_wait_front(cb_ones, 1);
    for (uint32_t r0 = 0; r0 < rows; r0 += batch) {
        const uint32_t n = (rows - r0) < batch ? (rows - r0) : batch;
        cb_wait_front(cb_sumsq_tiles, n);
        cb_reserve_back(cb_partial_sumsq, n);
        reconfig_data_format(cb_sumsq_tiles, cb_ones);  // srcA = in1 (sumsq), srcB = in0 (ones)
        pack_reconfig_data_format(cb_partial_sumsq);
        matmul_init(cb_ones, cb_sumsq_tiles, 1 /*transpose in1*/);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_ones, cb_sumsq_tiles, 0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_partial_sumsq);
        }
        tile_regs_release();
        cb_push_back(cb_partial_sumsq, n);
        cb_pop_front(cb_sumsq_tiles, n);
    }
}

// combine_rows_block: per owned row, column-reduce its row_gather gather tiles (reduce_tile COL, 1.0 scaler: row 0 =
// sum over the members), transpose row 0 -> column 0 in DEST, finalize column 0 -> out_cb (owner: cb_inv_owned, sent
// back to the members; row_partials: cb_inv_rms, every member combines every row itself).
template <uint32_t out_cb = cb_inv_owned>
ALWI void combine_rows_block(uint32_t owned, FinalizeInvRms finalize) {
    cb_wait_front(cb_gathered_sumsq, owned * row_gather);
    cb_wait_front(cb_ones, 1);
    for (uint32_t j = 0; j < owned; ++j) {
        cb_reserve_back(out_cb, 1);
        reconfig_data_format(cb_gathered_sumsq, cb_ones);  // REDUCE_COL: data -> srcA, scaler -> srcB
        pack_reconfig_data_format(out_cb);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered_sumsq, cb_ones, out_cb);
        tile_regs_acquire();
        for (uint32_t g = 0; g < row_gather; ++g) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gathered_sumsq, cb_ones, j * row_gather + g, 0, 0);
        }
        reduce_uninit();
        transpose_dest_init<DST_ACCUM_MODE>();
        transpose_dest<DST_ACCUM_MODE>(0);
        finalize(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
    cb_pop_front(cb_gathered_sumsq, owned * row_gather);
}

// owner block: row r of the block (rotation ofs) is owned by member (r + ofs) % k; this member combines its owned rows,
// and pops the gathered slots of the rows it does not own (uniform per block: lockstep slot addressing on all members).
ALWI void owner_combine_block(uint32_t rows, uint32_t member_idx, uint32_t ofs, FinalizeInvRms finalize) {
    constexpr uint32_t k = w_split_cores;
    const uint32_t r0 = (member_idx + k - ofs) % k;
    const uint32_t owned = r0 < rows ? (rows - r0 + k - 1) / k : 0;
    const uint32_t slots = ((rows + k - 1) / k) * row_gather;
    if (owned > 0) {
        combine_rows_block(owned, finalize);
    }
    const uint32_t unread = slots - owned * row_gather;
    if (unread > 0) {
        cb_wait_front(cb_gathered_sumsq, unread);
        cb_pop_front(cb_gathered_sumsq, unread);
    }
}

#ifdef AGRMS_HAS_GAMMA
// MulGammaRowBcast (base op): DEST[d] *= row-broadcast(cb_gamma page) via DEST-reuse; only row 0 of faces 0/1 read.
template <InputSpec In>
struct MulGammaRowBcast : DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA> {
    using Base = DestReuseBinary<BinaryFpuOp::Mul, In, DestReuseType::DEST_TO_SRCA>;
    static constexpr uint32_t GCb = In.cb_id;
    constexpr MulGammaRowBcast() noexcept = default;
    constexpr explicit MulGammaRowBcast(uint32_t base) noexcept : Base(base) {}

    static ALWI void init() {
        state_configure(GCb, __builtin_LINE());
        UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, GCb)));
        MATH((llk_math_eltwise_binary_init<
              EltwiseBinaryType::ELWMUL,
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
              EltwiseBinaryType::ELWMUL,
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
    TileAddressing::Offset);
using TimesGamma = MulGammaRowBcast<gamma_in>;

// prescale_gamma_block: x * gamma (gamma row-broadcast from its compact faces-0/1 row-0 pages) of the block's resident
// landing tiles -> cb_x_gamma (T_acc: no extra rounding against the fused chain, whose x * inv_rms stays in DEST).
// Runs between collapse (partial out) and combine (partials in): it hides under the partial exchange.
ALWI void prescale_gamma_block(uint32_t rows, uint32_t width, uint32_t gamma_pages) {
    cb_wait_front(cb_gamma, gamma_pages);  // resident slice (writer, once): a no-op after the first block
    constexpr auto x_in = input(
        cb_x_landing,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto g_in = input(cb_gamma, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row);
    using XTimesGamma = BinaryFpu<BinaryFpuOp::Mul, x_in, g_in>;
    eltwise_chain(
        IterationShape::grid(rows, width), XTimesGamma{StridedTileRange{0, w_pitch}}, PackTile<output(cb_x_gamma)>{});
}

// normalize_prescaled: (x * gamma) streamed from cb_x_gamma, times inv_rms (bcast col) -> cb_out_tiles
ALWI void normalize_prescaled(uint32_t rows, uint32_t width) {
    constexpr auto inv_rms_in =
        input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XgTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, input(cb_x_gamma), inv_rms_in>;
    eltwise_chain(IterationShape::grid(rows, width), XgTimesInvRms{}, PackTile<output(cb_out_tiles)>{});
}
#endif

// regime A: x = resident landing tiles of the block, grid(rows, width) Strided by w_pitch
ALWI void normalize_resident(uint32_t rows, uint32_t width) {
    constexpr auto x_in = input(
        cb_x_landing,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto inv_rms_in =
        input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_rms_in>;
#ifdef AGRMS_HAS_GAMMA
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{0, w_pitch}},
        TimesGamma{0},
        PackTile<output(cb_out_tiles)>{});
#else
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{0, w_pitch}},
        PackTile<output(cb_out_tiles)>{});
#endif
}

// regime A, one row segment: n tiles of block row i, x from x_cb at x (Strided: one row), gamma from column c0.
// Native placement normalizes a row in segments: the own slice from the input shard (cb_x_in), the rest from the
// landing; packed in column order into cb_out_tiles (the output shard).
template <uint32_t x_cb>
ALWI void normalize_segment(uint32_t i, StridedTileRange x, [[maybe_unused]] uint32_t c0, uint32_t n) {
    constexpr auto x_in = input(
        x_cb,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto inv_rms_in = input(
        cb_inv_rms,
        BroadcastDim::Col,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Col,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
    using XTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_rms_in>;
#ifdef AGRMS_HAS_GAMMA
    eltwise_chain(IterationShape::grid(1, n), XTimesInvRms{x, i}, TimesGamma{c0}, PackTile<output(cb_out_tiles)>{});
#else
    eltwise_chain(IterationShape::grid(1, n), XTimesInvRms{x, i}, PackTile<output(cb_out_tiles)>{});
#endif
}

// native WIDTH, ragged last member: pad the output row up to the shard pitch (pages never read)
ALWI void pad_out_row(uint32_t out_pad) {
    if (out_pad > 0) {
        cb_reserve_back(cb_out_tiles, out_pad);
        cb_push_back(cb_out_tiles, out_pad);
    }
}

// regime N bfp8, stage_row: unpack n landed tiles (Strided range x of x_cb) and pack them as bf16 into cb_x_stage,
// in landing-column order. The reader composes the dense row from these 2-byte datums (bfp8 datums cannot be shifted
// across shared-exponent groups).
template <uint32_t x_cb>
ALWI void stage_segment(StridedTileRange x, uint32_t n) {
    constexpr auto x_in = input(
        x_cb,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    eltwise_chain(IterationShape::grid(1, n), CopyTile<x_in>{x}, PackTile<output(cb_x_stage)>{});
}

// one landed row (block row i, local row i0 + i): landing, the own slice from the input shard when own_in_shard
ALWI void stage_row(uint32_t i0, uint32_t i, uint32_t own_c0) {
    if constexpr (own_in_shard) {
        if (own_c0 > 0) {
            stage_segment<cb_x_landing>(StridedTileRange{i * w_pitch, w_pitch}, own_c0);
        }
        stage_segment<cb_x_in>(StridedTileRange{(i0 + i) * Wt_lp, Wt_lp}, Wt_lp);
        if (own_c0 + Wt_lp < w_pitch) {
            stage_segment<cb_x_landing>(
                StridedTileRange{i * w_pitch + own_c0 + Wt_lp, w_pitch}, w_pitch - own_c0 - Wt_lp);
        }
    } else {
        stage_segment<cb_x_landing>(StridedTileRange{i * w_pitch, w_pitch}, w_pitch);
    }
}

// regime N: one window of n dense tiles [t0, t0 + n) of block row i, x streamed from cb_x_aligned
ALWI void normalize_streamed(uint32_t i, uint32_t t0, uint32_t n) {
    constexpr auto x_in = input(cb_x_aligned);  // PerTile wait / pop
    constexpr auto inv_rms_in = input(
        cb_inv_rms,
        BroadcastDim::Col,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Col,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
    using XTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_rms_in>;
#ifdef AGRMS_HAS_GAMMA
    eltwise_chain(IterationShape::grid(1, n), XTimesInvRms{0, i}, TimesGamma{t0}, PackTile<output(cb_out_tiles)>{});
#else
    eltwise_chain(IterationShape::grid(1, n), XTimesInvRms{0, i}, PackTile<output(cb_out_tiles)>{});
#endif
}

void kernel_main() {
#if defined(AGRMS_ABLATE_EMPTY) || defined(AGRMS_ABLATE_TRANSPORT)
    return;
#endif
    const uint32_t n_rows = get_arg_val<uint32_t>(0);
    const uint32_t core_w = get_arg_val<uint32_t>(1);  // regime A: member width; regime N: landing tiles per row
    const FinalizeInvRms finalize{get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};
    [[maybe_unused]] const uint32_t gamma_pages = get_arg_val<uint32_t>(4);  // cb_gamma pages incl. tail pages
    [[maybe_unused]] const uint32_t member_idx = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t own_c0 = get_arg_val<uint32_t>(6);   // own slice's first landing column
    [[maybe_unused]] const uint32_t out_pad = get_arg_val<uint32_t>(7);  // native: pad pages per output row

    compute_kernel_hw_startup(cb_x_landing, cb_x_landing, cb_out_tiles);
    if constexpr (own_in_shard) {
        cb_wait_front(cb_x_in, n_rows * Wt_lp);  // the resident input shard (published once by the reader)
    }

    for (uint32_t i0 = 0; i0 < n_rows; i0 += block_row_tiles) {
        const uint32_t rows = (n_rows - i0) < block_row_tiles ? (n_rows - i0) : block_row_tiles;
        {
            AGRMS_ZONE("AG_C_SUMSQ");
            sum_squares_block(i0, rows, core_w, own_c0);
        }
        {
            AGRMS_ZONE("AG_C_COMBINE");
            if constexpr (w_split_cores == 1) {
                collapse_sumsq_block<cb_inv_rms>(rows, finalize);
            } else if constexpr (combine_owner) {
                collapse_rows_to_row0(rows);
#ifdef AGRMS_HAS_GAMMA
                if constexpr (prescale) {
                    prescale_gamma_block(rows, core_w, gamma_pages);
                }
#endif
                owner_combine_block(rows, member_idx, i0 % w_split_cores, finalize);  // owners rotate per block
            } else {
                if constexpr (row_partials) {
                    collapse_rows_to_row0(rows);
                } else {
                    collapse_sumsq_block<cb_partial_sumsq>(rows, NoOp{});
                }
#ifdef AGRMS_HAS_GAMMA
                if constexpr (prescale) {
                    prescale_gamma_block(rows, core_w, gamma_pages);
                }
#endif
                if constexpr (row_partials) {
                    combine_rows_block<cb_inv_rms>(rows, finalize);
                } else {
                    combine_partials_block(rows, finalize);
                }
            }
            cb_wait_front(cb_inv_rms, rows);
        }
        AGRMS_ZONE("AG_C_NORM");
#ifdef AGRMS_HAS_GAMMA
        cb_wait_front(cb_gamma, gamma_pages);  // resident slice (writer, once): a no-op after the first block
#endif
        if constexpr (regime_n) {
            for (uint32_t i = 0; i < rows; ++i) {
                if constexpr (stage) {
                    stage_row(i0, i, own_c0);  // the reader composes row i from it while this core normalizes
                }
                for (uint32_t t0 = 0; t0 < out_w_tiles; t0 += out_unit) {
                    const uint32_t n = (out_w_tiles - t0) < out_unit ? (out_w_tiles - t0) : out_unit;
                    normalize_streamed(i, t0, n);
                }
            }
        } else if (own_in_shard) {
            // landing [0, own_c0) | input shard [own_c0, own_c0 + Wt_lp) | landing [own_c0 + Wt_lp, core_w)
            const uint32_t own_c1 = own_c0 + Wt_lp;
            for (uint32_t i = 0; i < rows; ++i) {
                if (own_c0 > 0) {
                    normalize_segment<cb_x_landing>(i, StridedTileRange{i * w_pitch, w_pitch}, 0, own_c0);
                }
                normalize_segment<cb_x_in>(i, StridedTileRange{(i0 + i) * Wt_lp, Wt_lp}, own_c0, Wt_lp);
                if (own_c1 < core_w) {
                    normalize_segment<cb_x_landing>(
                        i, StridedTileRange{i * w_pitch + own_c1, w_pitch}, own_c1, core_w - own_c1);
                }
            }
        } else if (out_pad > 0) {
            for (uint32_t i = 0; i < rows; ++i) {
                normalize_segment<cb_x_landing>(i, StridedTileRange{i * w_pitch, w_pitch}, 0, core_w);
                pad_out_row(out_pad);
            }
        } else if constexpr (prescale) {
#ifdef AGRMS_HAS_GAMMA
            normalize_prescaled(rows, core_w);
#endif
        } else {
            normalize_resident(rows, core_w);
        }
        cb_pop_front(cb_x_landing, rows * w_pitch);
        cb_pop_front(cb_inv_rms, rows);
    }
#ifdef AGRMS_HAS_GAMMA
    cb_pop_front(cb_gamma, gamma_pages);
#endif
    if constexpr (own_in_shard) {
        cb_pop_front(cb_x_in, n_rows * Wt_lp);
    }
}
