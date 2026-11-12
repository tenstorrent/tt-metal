// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (regime R1). Block schedule (op_design.md "Block schedule"):
//
//   for block:
//     tilize_x_block   [RM]     cb_x_rm -> cb_x              tilize<c>, one init/uninit per block
//     sumsq_block               cb_x -> cb_sumsq             Σ x² per row, packer L1 accumulation
//     rstd_block                cb_sumsq -> cb_rstd          reduce<SUM, ROW, AccumulateViaAdd>
//                                                            + post-op (·1/W, +eps, rsqrt)
//     for chunk:
//       normalize_chunk         cb_x · rstd(col) -> cb_out | cb_normed
//       scale_chunk   [gamma]   cb_normed · gamma(row) -> cb_out (TILE) | cb_normed in place (RM)
//       untilize_chunk [RM]     cb_normed -> cb_out_rm
//     release_block             pop cb_x / cb_rstd
//
// Raw calls, each with its reason:
//  - post_reduce_op SFPU calls (mul/add_unary_tile, rsqrt_tile): the reduce helper's documented
//    callback runs inside the helper's own DEST window, where a chain (which owns its own window)
//    cannot run (reduce_helpers_compute.hpp post_reduce_op contract).
//  - cb_x / cb_rstd / cb_sumsq wait/pop/reserve/push around chains configured with caller-managed
//    (None, None) policies: Strided/Offset tile addressing and L1 accumulation across several
//    chain calls require the caller to own those windows (chain.hpp TileAddressing notes).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t cb_x_rm = get_compile_time_arg_val(0);
constexpr uint32_t cb_x = get_compile_time_arg_val(1);
constexpr uint32_t cb_reduce_mask = get_compile_time_arg_val(2);
constexpr uint32_t cb_sumsq = get_compile_time_arg_val(3);
constexpr uint32_t cb_rstd = get_compile_time_arg_val(4);
constexpr uint32_t cb_gamma = get_compile_time_arg_val(5);
constexpr uint32_t cb_normed = get_compile_time_arg_val(6);
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_out_rm = get_compile_time_arg_val(8);

constexpr uint32_t block_rows = get_compile_time_arg_val(9);
constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(10);
constexpr uint32_t num_w_chunks = get_compile_time_arg_val(11);
constexpr uint32_t core_w_tiles = get_compile_time_arg_val(12);
constexpr uint32_t w_partial_elems = get_compile_time_arg_val(13);
constexpr bool has_gamma = get_compile_time_arg_val(14) != 0;
constexpr bool is_rm = get_compile_time_arg_val(15) != 0;
constexpr uint32_t inv_w_bits = get_compile_time_arg_val(16);
constexpr uint32_t eps_bits = get_compile_time_arg_val(17);

constexpr uint32_t row_stride_tiles = num_w_chunks * w_chunk_tiles;
constexpr bool has_partial = w_partial_elems != 0;
constexpr uint32_t n_full = core_w_tiles - (has_partial ? 1 : 0);  // tiles folded by L1 accumulation
constexpr uint32_t n_sq = (n_full > 0 ? 1 : 0) + (has_partial ? 1 : 0);
// The normalized chunk goes straight to the writer's CB when nothing else follows it (TILE, no gamma).
constexpr bool normalize_to_out = !is_rm && !has_gamma;
constexpr uint32_t cb_normalize_dst = normalize_to_out ? cb_out : cb_normed;

// ---- sumsq_block element types ----
using SquareX = BinaryFpu<
    BinaryFpuOp::Mul,
    input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset),
    input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset)>;
template <L1Accumulation Acc>
using PackSumsq = PackTile<output(
    cb_sumsq,
    ReservePolicy::None,
    PushPolicy::None,
    DataFormatReconfig::Enabled,
    TileAddressing::Direct,
    DestAccumulation::Disabled,
    Acc)>;
using PackSquaredTail = PackTile<output(cb_sumsq)>;

// ---- normalize / scale element types ----
using MulRstdCol = BinaryFpu<
    BinaryFpuOp::Mul,
    input(
        cb_x,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided),
    input(cb_rstd, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col)>;
using PackNormalized = PackTile<output(cb_normalize_dst, ReservePolicy::PerOuter, PushPolicy::PerOuter)>;

using MulGammaRow = BinaryFpu<
    BinaryFpuOp::Mul,
    input(cb_normed),
    input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Row)>;
// TILE: scaled chunk -> writer, one c-tile window per row. RM: in place on cb_normed, per-tile credit
// (the input tile is popped before the output slot is reserved, so a full cb_normed never deadlocks).
using PackScaled = std::conditional_t<
    is_rm,
    PackTile<output(cb_normed)>,
    PackTile<output(cb_out, ReservePolicy::PerOuter, PushPolicy::PerOuter)>>;

// One row-chunk (32 sticks x c tiles) per call. Across a block's back-to-back calls the LLK init and the
// register reconfig are paid once: only the InitOnly / InitAndUninit call reconfigures — re-running
// pack_reconfig_data_format after fast_tilize_init on a later call corrupts the fast-tilize pack setup
// (measured: W=4096, c=64, K=2 gave pcc ~0.5 until the follow-on calls stopped reconfiguring).
template <tilize_config::InitUninitMode Mode, class InDFB, class OutDFB>
ALWI void tilize_row_chunk(InDFB& in, OutDFB& out) {
    using tilize_config::InitUninitMode;
    using tilize_config::ReconfigureRegisterDatatypeMode;
    constexpr bool first_of_run = Mode == InitUninitMode::InitAndUninit || Mode == InitUninitMode::InitOnly;
    constexpr auto reconfig = first_of_run ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                           : ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::tilize<w_chunk_tiles, cb_x_rm, cb_x, Mode, tilize_config::WaitMode::WaitBlock, reconfig>(
        in, out, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_x, cb_rstd);

    DataflowBuffer x_rm(cb_x_rm);
    DataflowBuffer x(cb_x);
    DataflowBuffer mask(cb_reduce_mask);
    DataflowBuffer sumsq(cb_sumsq);
    DataflowBuffer rstd(cb_rstd);
    DataflowBuffer gamma(cb_gamma);
    DataflowBuffer normed(cb_normed);
    DataflowBuffer out(cb_out);
    DataflowBuffer out_rm(cb_out_rm);

    const uint32_t num_blocks = (num_tile_rows + block_rows - 1) / block_rows;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * block_rows;
        const uint32_t rows_this_block =
            (num_tile_rows - first_row < block_rows) ? (num_tile_rows - first_row) : block_rows;

        // ---- tilize_x_block (RM): one tilize init/uninit amortized over the block's row-chunks ----
        if constexpr (is_rm) {
            using tilize_config::InitUninitMode;
            const uint32_t calls = rows_this_block * num_w_chunks;
            for (uint32_t n = 0; n < calls; ++n) {
                if (calls == 1) {
                    tilize_row_chunk<InitUninitMode::InitAndUninit>(x_rm, x);
                } else if (n == 0) {
                    tilize_row_chunk<InitUninitMode::InitOnly>(x_rm, x);
                } else if (n + 1 == calls) {
                    tilize_row_chunk<InitUninitMode::UninitOnly>(x_rm, x);
                } else {
                    tilize_row_chunk<InitUninitMode::Neither>(x_rm, x);
                }
            }
        }

        // ---- sumsq_block: per row, Σ_t x_t² over the n_full whole tiles into ONE L1-accumulated tile,
        //      chunk by chunk as the reader lands them; then the squared partial tail tile (if any). ----
        for (uint32_t i = 0; i < rows_this_block; ++i) {
            const uint32_t row_base = i * row_stride_tiles;
            if constexpr (n_full > 0) {
                sumsq.reserve_back(1);
                for (uint32_t k = 0; k < num_w_chunks; ++k) {
                    const uint32_t lo = k * w_chunk_tiles;
                    if (lo >= n_full) {
                        break;
                    }
                    const uint32_t hi = (lo + w_chunk_tiles < n_full) ? (lo + w_chunk_tiles) : n_full;
                    x.wait_front(row_base + lo + w_chunk_tiles);  // cumulative: this row-chunk has landed
                    if (k == 0) {
                        eltwise_chain(
                            IterationShape::tiles(hi - lo),
                            SquareX{x, x, row_base + lo, row_base + lo},
                            PackSumsq<L1Accumulation::Enabled>{sumsq});
                    } else {
                        eltwise_chain(
                            IterationShape::tiles(hi - lo),
                            SquareX{x, x, row_base + lo, row_base + lo},
                            PackSumsq<L1Accumulation::AddToExisting>{sumsq});
                    }
                }
                sumsq.push_back(1);
            }
            if constexpr (has_partial) {
                const uint32_t tail = row_base + core_w_tiles - 1;
                x.wait_front(row_base + row_stride_tiles);
                eltwise_chain(IterationShape::one_tile(), SquareX{x, x, tail, tail}, PackSquaredTail{sumsq});
            }
        }

        // ---- rstd_block: one reduce over (rows_this_block × n_sq); the masked tail is folded in by the
        //      helper's AccumulateViaAdd partial path; post-op turns Σx² into 1/sqrt(Σx²/W + eps). ----
        constexpr auto partial = has_partial ? ReducePartialScaler::only_partial() : ReducePartialScaler::none();
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sumsq,
            cb_reduce_mask,
            cb_rstd,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ReduceAlgorithm::AccumulateViaAdd,
            ReduceWithinTile::Collapse>(
            sumsq,
            mask,
            rstd,
            ReduceInputBlockShape::of(rows_this_block, n_sq),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            [](uint32_t dst_idx) {
                binop_with_scalar_tile_init();
                mul_unary_tile(dst_idx, inv_w_bits);
                add_unary_tile(dst_idx, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            },
            partial);

        // ---- normalize / scale / untilize, chunk by chunk ----
        x.wait_front(rows_this_block * row_stride_tiles);
        rstd.wait_front(rows_this_block);
        for (uint32_t k = 0; k < num_w_chunks; ++k) {
            eltwise_chain(
                IterationShape::grid(rows_this_block, w_chunk_tiles),
                MulRstdCol{x, rstd, StridedTileRange{k * w_chunk_tiles, row_stride_tiles}},
                PackNormalized{normalize_to_out ? out : normed});

            if constexpr (has_gamma) {
                eltwise_chain(
                    IterationShape::grid(rows_this_block, w_chunk_tiles),
                    MulGammaRow{normed, gamma},
                    PackScaled{is_rm ? normed : out});
            }

            if constexpr (is_rm) {
                compute_kernel_lib::untilize<w_chunk_tiles, cb_normed, cb_out_rm>(normed, out_rm, rows_this_block);
            }
        }

        // ---- release_block ----
        x.pop_front(rows_this_block * row_stride_tiles);
        rstd.pop_front(rows_this_block);
    }

    if constexpr (has_partial) {
        mask.wait_front(1);
        mask.pop_front(1);
    }
}
