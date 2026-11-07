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
ALWI void sum_squares_block(uint32_t rows, uint32_t full_w) {
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
                mul_tiles(cb_x_landing, cb_x_landing, base + j, base + j, i);
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
    const uint32_t n_rows = get_arg_val<uint32_t>(0);
    const uint32_t core_w = get_arg_val<uint32_t>(1);  // regime A: member width; regime N: landing tiles per row
    const FinalizeInvRms finalize{get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};
    [[maybe_unused]] const uint32_t gamma_pages = get_arg_val<uint32_t>(4);  // cb_gamma pages incl. tail pages

    compute_kernel_hw_startup(cb_x_landing, cb_x_landing, cb_out_tiles);

    for (uint32_t i0 = 0; i0 < n_rows; i0 += block_row_tiles) {
        const uint32_t rows = (n_rows - i0) < block_row_tiles ? (n_rows - i0) : block_row_tiles;
        sum_squares_block(rows, core_w);
        if constexpr (w_split_cores == 1) {
            collapse_sumsq_block<cb_inv_rms>(rows, finalize);
        } else {
            collapse_sumsq_block<cb_partial_sumsq>(rows, NoOp{});
            combine_partials_block(rows, finalize);
        }
        cb_wait_front(cb_inv_rms, rows);
#ifdef AGRMS_HAS_GAMMA
        cb_wait_front(cb_gamma, gamma_pages);  // resident slice (writer, once): a no-op after the first block
#endif
        if constexpr (regime_n) {
            for (uint32_t i = 0; i < rows; ++i) {
                for (uint32_t t0 = 0; t0 < out_w_tiles; t0 += out_unit) {
                    const uint32_t n = (out_w_tiles - t0) < out_unit ? (out_w_tiles - t0) : out_unit;
                    normalize_streamed(i, t0, n);
                }
            }
        } else {
            normalize_resident(rows, core_w);
        }
        cb_pop_front(cb_x_landing, rows * w_pitch);
        cb_pop_front(cb_inv_rms, rows);
    }
#ifdef AGRMS_HAS_GAMMA
    cb_pop_front(cb_gamma, gamma_pages);
#endif
}
