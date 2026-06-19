// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm ROW_MAJOR compute kernel (tilize-wrapped, row-parallel).
//
// All math runs on tiles. Row-major sticks arrive via cb_rm_in; this kernel
// tilizes them, computes the per-stick RMS normalization on tiles, untilizes
// the result, and hands row-major sticks to the writer via cb_rm_out.
//
// Per 32-stick tile-block (Wt_padded = num_chunks * reduce_block tiles wide):
//   TILIZE    cb_rm_in chunks -> cb_input_resident (held resident across passes)
//   PASS 1    sum of squares over the resident block, chunked by reduce_block
//             (square -> reduce<SUM,REDUCE_ROW,Accumulate>) -> cb_partial_sumsq
//   FINALIZE  rsqrt(sum * inv_W + eps) -> cb_recip_rms   (inv_W = 1/W, true count)
//   PASS 2    per chunk: [tilize gamma chunk ->] x * recip (Col bcast)
//             [* gamma (Row bcast)] -> cb_out_tiled -> untilize -> cb_rm_out
//
// Non-aligned correctness is handled in the dataflow kernels: the reader zeroes
// W-padding columns (so they add 0 to the per-stick Sum(x^2)) and zeroes/skips
// H-padding sticks; the writer writes only valid columns/sticks. inv_W carries
// the TRUE element count, so the RMS denominator counts only valid elements.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(5);
    constexpr uint32_t cb_squared = get_compile_time_arg_val(6);
    constexpr uint32_t cb_partial_sumsq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_recip_rms = get_compile_time_arg_val(8);
    constexpr uint32_t cb_normalized = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_tiled = get_compile_time_arg_val(10);
    constexpr uint32_t reduce_block = get_compile_time_arg_val(11);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(13);
    constexpr uint32_t gamma_is_tile = get_compile_time_arg_val(14);  // gamma.layout == TILE
    constexpr uint32_t inv_W_bits = get_compile_time_arg_val(15);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(16);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    using ckl::Accumulate;
    using ckl::BinaryDataFormatReconfig;
    using ckl::BinaryFpu;
    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::CopyTile;
    using ckl::Dst;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::OperandKind;
    using ckl::OutputLifecycle;
    using ckl::PackTile;
    using ckl::ReduceInputBlockShape;
    using ckl::ReduceInputPolicy;
    using ckl::TileOffset;

    namespace tcfg = ckl::tilize_config;
    namespace ucfg = ckl::untilize_config;

    compute_kernel_hw_startup(cb_rm_in, cb_input_resident);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ---------- TILIZE: cb_rm_in chunks -> cb_input_resident (resident) ----------
        for (uint32_t c = 0; c < num_chunks; ++c) {
            ckl::tilize<reduce_block, cb_rm_in, cb_input_resident>(1);
        }

        // ---------- PASS 1: sum of squares over the resident block ----------
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t base = c * reduce_block;

            ckl::eltwise_chain(
                EltwiseShape::tiles(reduce_block),
                BinaryFpu<
                    cb_input_resident,
                    cb_input_resident,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    InputLifecycle::HeldBulk,
                    InputLifecycle::HeldBulk,
                    BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Block,
                    TileOffset::Set,
                    TileOffset::Set>{base, base},
                PackTile<cb_squared, OutputLifecycle::Streaming>{});

            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
                cb_squared,
                cb_scaler,
                cb_partial_sumsq,
                ReduceInputBlockShape::of(1, reduce_block),
                ckl::ReduceInputMemoryLayout::contiguous(),
                Accumulate(cb_partial_sumsq, c));
        }

        // ---------- FINALIZE: rsqrt(sum * inv_W + eps) ----------
        ckl::eltwise_chain(
            EltwiseShape::tiles(1),
            CopyTile<cb_partial_sumsq, Dst::D0, InputLifecycle::Streaming>{},
            ckl::MulUnary<>{inv_W_bits},
            ckl::AddUnary<>{eps_bits},
            ckl::Rsqrt<>{},
            PackTile<cb_recip_rms, OutputLifecycle::Streaming>{});

        // ---------- PASS 2: normalize, per chunk, untilize -> cb_rm_out ----------
        for (uint32_t c = 0; c < num_chunks; ++c) {
            if constexpr (has_gamma) {
                // ROW_MAJOR gamma: tilize this chunk's sticks (row 0 valid) ->
                // cb_gamma_tiled. TILE gamma: the reader already filled
                // cb_gamma_tiled with column tiles directly (no tilize needed).
                if constexpr (!gamma_is_tile) {
                    ckl::tilize<reduce_block, cb_gamma_rm, cb_gamma_tiled>(1);
                }

                // x * recip (Col bcast) -> cb_normalized
                ckl::eltwise_chain(
                    EltwiseShape::tiles(reduce_block),
                    BinaryFpu<
                        cb_input_resident,
                        cb_recip_rms,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    PackTile<cb_normalized, OutputLifecycle::Streaming>{});

                // normalized * gamma (Row bcast) -> cb_out_tiled
                ckl::eltwise_chain(
                    EltwiseShape::tiles(reduce_block),
                    BinaryFpu<
                        cb_normalized,
                        cb_gamma_tiled,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Row,
                        InputLifecycle::Streaming,
                        InputLifecycle::Streaming,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    PackTile<cb_out_tiled, OutputLifecycle::Streaming>{});
            } else {
                // x * recip (Col bcast) -> cb_out_tiled
                ckl::eltwise_chain(
                    EltwiseShape::tiles(reduce_block),
                    BinaryFpu<
                        cb_input_resident,
                        cb_recip_rms,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    PackTile<cb_out_tiled, OutputLifecycle::Streaming>{});
            }

            // untilize this chunk -> cb_rm_out (writer drains)
            ckl::untilize<reduce_block, cb_out_tiled, cb_rm_out>(1);
        }

        // recip was held across the PASS-2 chunk loop; free this block's tile.
        cb_pop_front(cb_recip_rms, 1);
    }
}
