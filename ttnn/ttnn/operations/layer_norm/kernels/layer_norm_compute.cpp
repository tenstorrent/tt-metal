// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute: one NEO of C. It sees its own slice of every STRIDED ring, so this reads as the
// single-threaded kernel over this NEO's `rounds` tile-rows, driven in blocks of `block_tile_rows` rows.
//
// Per block of Rb rows (op_design.md, Block schedule), every step a kernel-library helper:
//   row_sum   Σ_w (x_w + x_w) accumulated in DEST, one tile per row          input_block -> row_partial
//   mean      reduce_row(partial) · 1/(2W)                                     row_partial -> row_stat
//   centre    c = x − mean (mean broadcast down the columns)                   input_block -> centred (pops x)
//   sq_sum    Σ_w c_w · c_w accumulated in DEST                                centred -> row_partial
//   var       reduce_row(partial) · 1/W                                        row_partial -> row_stat
//   rstd      rsqrt(var + ε), ε a scalar broadcast                             row_stat -> row_partial
//   affine    (c · rstd) · γ_rows + β_rows (rstd broadcast down the columns)   centred, row_partial -> output_tiles
// Prologue: γ and β (row 0 of `weight` / `bias`, rule R8) broadcast down the rows into full tiles, once per
// kernel, so the affine step can apply them through DEST reuse (which takes no broadcast on Quasar).
//
// Departure from the design's realisation (op_design.md ops 7 and 8; the algorithm and its order are kept):
// the design broadcasts the variance with UnaryBcast<Col> and adds a full ε tile through DEST reuse. On
// Quasar, unary_bcast of a 32-bit operand is refused (tt_metal/hw/inc/api/compute/bcast.h, Quasar path:
// LLK_ASSERT "32-bit unary broadcast (unpack-to-dest) not supported on Quasar"), and row_stat is Float32 under
// fp32 accumulation. So ε is added by the FPU with a scalar broadcast of srcB (one valid element in the ε
// tile, no whole-tile fill on the data-movement side), rstd stays a column-0 tile, and the affine step
// broadcasts it down the columns as the FPU's srcB; γ is then expanded like β.
//
// Raw-LLK call (the one in this kernel): `alu_source_formats(a, b)` = llk_math_hw_configure, before every FPU
// chain. On Quasar the FPU reads its source formats from the ALU format registers, programmed once by
// compute_kernel_hw_startup. This kernel mixes a bfloat16 input with Float32 intermediates under fp32
// accumulation; the call is what compute_kernel_hw_startup itself issues.
// Inside one chain the formats cannot change, so the affine chain's three FPU operations share one format: the
// γ/β rings are in the intermediate format, like DEST.
//
// pack_init(<ring>) before every helper whose output ring differs from the previous one's. The
// reduce helper issues its own pack_init.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "layer_norm_rows.hpp"

namespace ckl = compute_kernel_lib;
using ckl::BinaryFpu;
using ckl::BinaryFpuOp;
using ckl::BroadcastDim;
using ckl::DataFormatReconfig;
using ckl::DestAccumulation;
using ckl::DestReuseBinary;
using ckl::DestReuseType;
using ckl::Dst;
using ckl::input;
using ckl::InputTileMapping;
using ckl::IterationShape;
using ckl::output;
using ckl::PackTile;
using ckl::PopPolicy;
using ckl::PushPolicy;
using ckl::ReservePolicy;
using ckl::TileAddressing;
using ckl::WaitPolicy;

// Point the ALU source formats at the two operands of the next FPU chain (see the header).
ALWI void alu_source_formats(uint32_t srca_dfb, uint32_t srcb_dfb) {
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(srca_dfb, srcb_dfb)));
}

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t C = get_arg(args::compute_threads);
    const uint32_t rounds = layer_norm_rows::rounds(num_rows, C);  // the same for every NEO of the cluster

    DataflowBuffer input_block(dfb::input_block);
    DataflowBuffer affine_stream(dfb::affine_stream);
    DataflowBuffer gamma_rows(dfb::gamma_rows);
    DataflowBuffer beta_rows(dfb::beta_rows);
    DataflowBuffer centred(dfb::centred);
    DataflowBuffer row_partial(dfb::row_partial);
    DataflowBuffer row_stat(dfb::row_stat);
    DataflowBuffer output_tiles(dfb::output_tiles);
    DataflowBuffer mean_scaler(dfb::mean_scaler);
    DataflowBuffer var_scaler(dfb::var_scaler);
    DataflowBuffer epsilon(dfb::epsilon);

    compute_kernel_hw_startup(dfb::input_block, dfb::affine_stream, dfb::output_tiles);

    // ---- prologue: expand γ, then β (row 0 of each tile broadcast down its rows), resident for the kernel ----
    alu_source_formats(dfb::affine_stream, dfb::affine_stream);
    constexpr auto next_affine = input(dfb::affine_stream, WaitPolicy::PerTile, PopPolicy::PerTile);
    pack_init(dfb::gamma_rows);
    ckl::eltwise_chain(
        IterationShape::tiles(Wt),
        ckl::UnaryBcast<BroadcastDim::Row, next_affine>{affine_stream},
        PackTile<output(dfb::gamma_rows)>{gamma_rows});
    pack_init(dfb::beta_rows);
    ckl::eltwise_chain(
        IterationShape::tiles(Wt),
        ckl::UnaryBcast<BroadcastDim::Row, next_affine>{affine_stream},
        PackTile<output(dfb::beta_rows)>{beta_rows});

    // ---- operand specs of the block operations ----
    // The input block grows tile by tile as the reader lands it (Cumulative): row_sum runs inside the read.
    constexpr auto x_growing =
        input(dfb::input_block, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block);
    constexpr auto x_next = input(dfb::input_block, WaitPolicy::PerTile, PopPolicy::PerTile);
    constexpr auto c_resident = input(dfb::centred, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block);
    constexpr auto c_next = input(dfb::centred, WaitPolicy::PerTile, PopPolicy::PerTile);
    // One per-row tile, streamed: waited at the start of each grid row, popped at its end.
    constexpr auto mean_per_row =
        input(dfb::row_stat, BroadcastDim::Col, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Col);
    // rstd is the third FIFO use of row_partial in a block (after 2·Σx and Σc², each popped by its reduce).
    constexpr auto rstd_per_row =
        input(dfb::row_partial, BroadcastDim::Col, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Col);
    constexpr auto var_next = input(dfb::row_stat, WaitPolicy::PerTile, PopPolicy::PerTile);
    constexpr auto eps_scalar =
        input(dfb::epsilon, BroadcastDim::Scalar, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Scalar);
    constexpr auto gamma_row = input(dfb::gamma_rows, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row);
    constexpr auto beta_row = input(dfb::beta_rows, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row);
    constexpr auto partial_per_row = output(
        dfb::row_partial,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow);

    for (uint32_t block_start = 0; block_start < rounds; block_start += block_tile_rows) {
        // A ragged last block passes a smaller extent into the same operations.
        const uint32_t Rb = (rounds - block_start < block_tile_rows) ? rounds - block_start : block_tile_rows;
        const auto block = IterationShape::grid(Rb, Wt);

        // row_sum: DEST[row] += x + x over the row's Wt tiles.
        alu_source_formats(dfb::input_block, dfb::input_block);
        pack_init(dfb::row_partial);
        ckl::eltwise_chain(
            block,
            BinaryFpu<BinaryFpuOp::Add, x_growing, x_growing, Dst::D0, DestAccumulation::PerRow>{
                input_block, input_block},
            PackTile<partial_per_row>{row_partial});

        // mean = reduce_row(2Σx) · 1/(2W), column 0 of one tile per row.
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::row_partial, dfb::mean_scaler, dfb::row_stat>(
            row_partial, mean_scaler, row_stat, ckl::ReduceInputBlockShape::of(Rb, 1));

        // centre: c = x − mean; frees the input block for the reader's next one.
        alu_source_formats(dfb::input_block, dfb::row_stat);
        pack_init(dfb::centred);
        ckl::eltwise_chain(
            block,
            BinaryFpu<BinaryFpuOp::Sub, x_next, mean_per_row>{input_block, row_stat},
            PackTile<output(dfb::centred)>{centred});

        // sq_sum: DEST[row] += c · c over the row's Wt tiles (the centred block stays resident).
        alu_source_formats(dfb::centred, dfb::centred);
        pack_init(dfb::row_partial);
        ckl::eltwise_chain(
            block,
            BinaryFpu<BinaryFpuOp::Mul, c_resident, c_resident, Dst::D0, DestAccumulation::PerRow>{centred, centred},
            PackTile<partial_per_row>{row_partial});

        // var = reduce_row(Σc²) · 1/W (biased).
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::row_partial, dfb::var_scaler, dfb::row_stat>(
            row_partial, var_scaler, row_stat, ckl::ReduceInputBlockShape::of(Rb, 1));

        // rstd = rsqrt(var + ε), valid in column 0; into row_partial, which var's reduce has emptied.
        alu_source_formats(dfb::row_stat, dfb::epsilon);
        pack_init(dfb::row_partial);
        ckl::eltwise_chain(
            IterationShape::tiles(Rb),
            BinaryFpu<BinaryFpuOp::Add, var_next, eps_scalar>{row_stat, epsilon},
            ckl::Rsqrt<>{},
            PackTile<output(dfb::row_partial)>{row_partial});

        // affine: out = (c · rstd) · γ + β; pops the centred block and the rstd tiles.
        alu_source_formats(dfb::centred, dfb::row_partial);
        pack_init(dfb::output_tiles);
        ckl::eltwise_chain(
            block,
            BinaryFpu<BinaryFpuOp::Mul, c_next, rstd_per_row>{centred, row_partial},
            DestReuseBinary<BinaryFpuOp::Mul, gamma_row, DestReuseType::DEST_TO_SRCA>{gamma_rows},
            DestReuseBinary<BinaryFpuOp::Add, beta_row, DestReuseType::DEST_TO_SRCA>{beta_rows},
            PackTile<output(dfb::output_tiles)>{output_tiles});
    }

    // Kernel-lifetime rings must be empty when the kernel exits.
    gamma_rows.pop_front(Wt);
    beta_rows.pop_front(Wt);
    mean_scaler.pop_front(1);
    var_scaler.pop_front(1);
    epsilon.pop_front(1);
}
