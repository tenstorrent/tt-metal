// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC0/1/2). Block operations (op_design.md -> Block Operation Realization):
//   tilize_x_block      (RM)   compute_kernel_lib::tilize<C>        cb_x_rm -> cb_x
//   sum_squares_block          eltwise_chain BinaryFpu(x*x, PerRow DEST accumulation) -> cb_sumsq_tiles
//   collapse_sumsq_block       reduce<SUM, ROW, AccumulateViaAdd, Collapse>  -> cb_inv_rms (k==1, + finalize)
//                                                                           -> cb_partial_sumsq (k>1)
//   combine_partials_block     reduce<SUM, ROW, AccumulateViaAdd, Skip>(rows x k) + finalize -> cb_inv_rms
//   normalize_block            eltwise_chain BinaryFpu(x * inv_rms, bcast Col) [-> DestReuse * gamma] -> cb_out
//   untilize_out_block  (RM)   compute_kernel_lib::untilize<C>      cb_out -> cb_out_rm
//
// Raw API (documented exception): the finalize post-op (mul 1/W, add eps, rsqrt) runs on DST inside the
// reduce() helper's post_reduce_op hook — reduce() exposes that hook as a uint32_t dst callable, not a chain,
// so an eltwise_chain Rsqrt would need the reduced tile packed to L1 and re-read (extra CB + pass).
//
// cb_x indexing: a tile-row occupies x_row_pitch pages in cb_x (uniform across the program; the last W member
// may own fewer tiles, its gap pages are never read), so cb_x is addressed Strided{base, x_row_pitch}.
// x_block_depth == 2 is realized as two one-block CBs (cb_x / cb_x_alt, block parity) rather than one ring of
// two blocks: indexed CB reads cannot wrap the ring, and the combine lookahead reads block b+1 while block b
// is still resident.
//
// Combine lookahead (k > 1, x_block_depth == 2): pass 1 + collapse of block b+1 run before combine(b), so they
// overlap the partial all-gather of block b (the only stage that waits on other Tensix cores).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef RMS_RM
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif
#include "rms_norm_common.hpp"

using namespace compute_kernel_lib;
using namespace rms_norm_cb;

constexpr uint32_t w_split_cores = get_compile_time_arg_val(0);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(1);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(2);
constexpr uint32_t x_block_depth = get_compile_time_arg_val(3);
constexpr bool combine_lookahead = get_compile_time_arg_val(4) != 0;

// rsqrt(sum * (1/W) + eps) on DST[dst] — the reduce() post_reduce_op.
struct FinalizeInvRms {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
    }
};

// ---- sum_squares_block: per tile-row, sum_j x[r][j]^2 accumulated in DEST, one fp32 tile per tile-row ----
template <uint32_t cb_x_blk>
ALWI void sum_squares_block(uint32_t rows, uint32_t core_w_tiles) {
    cb_wait_front(cb_x_blk, rows * x_row_pitch);
    constexpr auto x_in = input(
        cb_x_blk,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    using Square = BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D0, DestAccumulation::PerRow>;
    using PackRowSum = PackTile<output(
        cb_sumsq_tiles,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;
    eltwise_chain(
        IterationShape::grid(rows, core_w_tiles),
        Square{StridedTileRange{0, x_row_pitch}, StridedTileRange{0, x_row_pitch}},
        PackRowSum{});
}

// ---- collapse_sumsq_block: within-tile row collapse (column 0 = row sum) [+ finalize] ----
template <uint32_t out_cb, typename PostOp>
ALWI void collapse_sumsq_block(uint32_t rows, PostOp post_op) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq_tiles,
        cb_x,  // scaler slot: never read on this path (no partial tile, default reload) — must be distinct
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(
        ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op);
}

// ---- combine_partials_block: sum the k gathered column-0 partials per tile-row + finalize ----
ALWI void combine_partials_block(uint32_t rows, FinalizeInvRms finalize) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered_sumsq,
        cb_x,  // scaler slot: never read (see above)
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

// ---- normalize_block: x * inv_rms (bcast Col) [* expanded gamma via DEST reuse] -> cb_out ----
// base_col: first tile column of the window (RM: chunk start); width: tiles walked per tile-row.
template <uint32_t cb_x_blk>
ALWI void normalize_window(uint32_t rows, uint32_t base_col, uint32_t width) {
    constexpr auto x_in = input(
        cb_x_blk,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    constexpr auto inv_rms_in =
        input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col);
    using XTimesInvRms = BinaryFpu<BinaryFpuOp::Mul, x_in, inv_rms_in>;
#ifdef RMS_HAS_GAMMA
    constexpr auto gamma_in = input(
        cb_gamma,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Row,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset);
    using TimesGamma = DestReuseBinary<BinaryFpuOp::Mul, gamma_in, DestReuseType::DEST_TO_SRCA>;
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{base_col, x_row_pitch}},
        TimesGamma{base_col},
        PackTile<output(cb_out)>{});
#else
    eltwise_chain(
        IterationShape::grid(rows, width),
        XTimesInvRms{StridedTileRange{base_col, x_row_pitch}},
        PackTile<output(cb_out)>{});
#endif
}

#ifdef RMS_RM
// One tilize call per tile-row: chunks_this blocks of width C (each waits C stick-piece pages of cb_x_rm and
// pushes C tiles into cb_x). The data-format reconfig runs only on the call that also inits: a reconfig issued
// after fast_tilize_init clobbers the fast-tilize unpack config (later tile-rows come out with rows 1..31 zero).
template <tilize_config::InitUninitMode mode>
ALWI void tilize_row(uint32_t chunks_this) {
    constexpr bool inits =
        mode == tilize_config::InitUninitMode::InitAndUninit || mode == tilize_config::InitUninitMode::InitOnly;
    tilize<
        rm_chunk_w_tiles,
        cb_x_rm,
        cb_x,
        mode,
        tilize_config::WaitMode::WaitBlock,
        inits ? tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
              : tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        tilize_config::Fp32Mode::Fast>(chunks_this);
}

// tilize_x_block: tilize init/uninit once per block (InitOnly / Neither / UninitOnly across tile-rows); each
// tile-row is padded to x_row_pitch pages (gap pages pushed unwritten — only the last W member has a gap).
ALWI void tilize_x_block(uint32_t rows, uint32_t chunks_this) {
    const uint32_t gap = x_row_pitch - chunks_this * rm_chunk_w_tiles;
    for (uint32_t r = 0; r < rows; ++r) {
        if (rows == 1) {
            tilize_row<tilize_config::InitUninitMode::InitAndUninit>(chunks_this);
        } else if (r == 0) {
            tilize_row<tilize_config::InitUninitMode::InitOnly>(chunks_this);
        } else if (r + 1 == rows) {
            tilize_row<tilize_config::InitUninitMode::UninitOnly>(chunks_this);
        } else {
            tilize_row<tilize_config::InitUninitMode::Neither>(chunks_this);
        }
        if (gap > 0) {
            cb_reserve_back(cb_x, gap);
            cb_push_back(cb_x, gap);
        }
    }
}
#endif

// pass 1 + collapse of one block; k == 1 finalizes into cb_inv_rms, k > 1 leaves the raw partial.
template <uint32_t cb_x_blk>
ALWI void reduce_front(uint32_t rows, uint32_t core_w_tiles, FinalizeInvRms finalize) {
    sum_squares_block<cb_x_blk>(rows, core_w_tiles);
    if constexpr (w_split_cores == 1) {
        collapse_sumsq_block<cb_inv_rms>(rows, finalize);
    } else {
        collapse_sumsq_block<cb_partial_sumsq>(rows, NoOp{});
    }
}

template <uint32_t cb_x_blk>
ALWI void apply_back(uint32_t rows, uint32_t core_w_tiles, uint32_t chunks_this) {
    cb_wait_front(cb_inv_rms, rows);
#ifdef RMS_RM
    for (uint32_t c = 0; c < chunks_this; ++c) {
        normalize_window<cb_x_blk>(rows, c * rm_chunk_w_tiles, rm_chunk_w_tiles);
        // untilize init/uninit per chunk: the normalize chain between chunks reprograms unpack/math/pack.
        untilize<rm_chunk_w_tiles, cb_out, cb_out_rm>(rows);
    }
#else
    normalize_window<cb_x_blk>(rows, 0, core_w_tiles);
#endif
    cb_pop_front(cb_x_blk, rows * x_row_pitch);
    cb_pop_front(cb_inv_rms, rows);
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(4);
    const FinalizeInvRms finalize{get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6)};

#ifdef RMS_RM
    compute_kernel_hw_startup(cb_x_rm, cb_x_rm, cb_x);
#else
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
#endif

#ifdef RMS_HAS_GAMMA
#ifdef RMS_RM
    const uint32_t gamma_tiles = chunks_this * rm_chunk_w_tiles;
#else
    const uint32_t gamma_tiles = core_w_tiles;
#endif
    cb_wait_front(cb_gamma, gamma_tiles);
#endif

    auto rows_of = [&](uint32_t blk) { return (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles; };
    auto use_alt = [&](uint32_t blk) { return x_block_depth == 2 && (blk & 1); };

#ifdef RMS_RM
    // RM: x_block_depth == 1 (compute is cb_x's producer): strictly sequential per block.
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = rows_of(blk);
        tilize_x_block(rows, chunks_this);
        reduce_front<cb_x>(rows, core_w_tiles, finalize);
        if constexpr (w_split_cores > 1) {
            combine_partials_block(rows, finalize);
        }
        apply_back<cb_x>(rows, core_w_tiles, chunks_this);
    }
#else
    if constexpr (w_split_cores > 1 && combine_lookahead) {
        reduce_front<cb_x>(rows_of(0), core_w_tiles, finalize);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t rows = rows_of(blk);
            if (blk + 1 < num_blocks) {
                if (use_alt(blk + 1)) {
                    reduce_front<cb_x_alt>(rows_of(blk + 1), core_w_tiles, finalize);
                } else {
                    reduce_front<cb_x>(rows_of(blk + 1), core_w_tiles, finalize);
                }
            }
            combine_partials_block(rows, finalize);
            if (use_alt(blk)) {
                apply_back<cb_x_alt>(rows, core_w_tiles, chunks_this);
            } else {
                apply_back<cb_x>(rows, core_w_tiles, chunks_this);
            }
        }
    } else {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t rows = rows_of(blk);
            if (use_alt(blk)) {
                reduce_front<cb_x_alt>(rows, core_w_tiles, finalize);
                if constexpr (w_split_cores > 1) {
                    combine_partials_block(rows, finalize);
                }
                apply_back<cb_x_alt>(rows, core_w_tiles, chunks_this);
            } else {
                reduce_front<cb_x>(rows, core_w_tiles, finalize);
                if constexpr (w_split_cores > 1) {
                    combine_partials_block(rows, finalize);
                }
                apply_back<cb_x>(rows, core_w_tiles, chunks_this);
            }
        }
    }
#endif

#ifdef RMS_HAS_GAMMA
    cb_pop_front(cb_gamma, gamma_tiles);
#endif
}
