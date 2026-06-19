// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute kernel (shared by Regime A and Regime B).
//
// Per tile-row block (Phase 0: bh = 1 tile-row, Wt_s tiles wide):
//   PASS 1   sum of squares over the resident shard, chunked by REDUCE_BLOCK
//            (square -> reduce<SUM,REDUCE_ROW,Accumulate>) -> cb_partial_sumsq
//   [B]      combine the K gathered partials (plain elementwise add) -> cb_partial_sumsq
//   FINALIZE rsqrt(sum * inv_W + eps) -> cb_recip_rms
//   PASS 2   normalize x * recip (Col bcast) [* gamma (Row bcast)] -> cb_output
//
// Helpers own all CB and DST ops; manual CB ops only appear between helper calls.
// Caller (this kernel) owns the single compute_kernel_hw_startup at boot.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partials_gathered = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output = get_compile_time_arg_val(4);
    constexpr uint32_t cb_squared = get_compile_time_arg_val(5);
    constexpr uint32_t cb_partial_sumsq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_recip_rms = get_compile_time_arg_val(7);
    constexpr uint32_t cb_normalized = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);  // tiles per shard along W
    constexpr uint32_t reduce_block = get_compile_time_arg_val(10);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(11);
    constexpr uint32_t inv_W_bits = get_compile_time_arg_val(12);    // fp32 bits of 1/W
    constexpr uint32_t eps_bits = get_compile_time_arg_val(13);      // fp32 bits of epsilon
    constexpr uint32_t num_partials = get_compile_time_arg_val(14);  // K (1 in Regime A)

    // ---- runtime args ----
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

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
    using ckl::PackTileReconfig;
    using ckl::ReduceInputBlockShape;
    using ckl::ReduceInputPolicy;
    using ckl::TileOffset;

    compute_kernel_hw_startup(cb_input_resident, cb_scaler, cb_output);

    constexpr uint32_t num_chunks = (Wt + reduce_block - 1) / reduce_block;

    for (uint32_t row = 0; row < num_rows; ++row) {
        // ---------- PASS 1: sum of squares over the resident shard ----------
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t base = c * reduce_block;
            const uint32_t cw = (base + reduce_block <= Wt) ? reduce_block : (Wt - base);

            // square resident[base .. base+cw) -> cb_squared (no pop of resident)
            ckl::eltwise_chain(
                EltwiseShape::tiles(cw),
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

            // reduce-accumulate the squared chunk into cb_partial_sumsq
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
                cb_squared,
                cb_scaler,
                cb_partial_sumsq,
                ReduceInputBlockShape::of(1, cw),
                ckl::ReduceInputMemoryLayout::contiguous(),
                Accumulate(cb_partial_sumsq, c));
        }

        // ---------- (Regime B) combine K gathered partials ----------
        // cb_partials_gathered holds K column-tiles (one local partial per W-shard, gathered
        // over the mcast rectangle). Their plain elementwise sum is the global Sum(x^2) over
        // the full W. Stream them: copy slot 0, then in-place add the remaining K-1 slots.
        // The reader already popped this core's local cb_partial_sumsq, so it is free to reuse.
        if constexpr (num_partials > 1) {
            ckl::copy<cb_partials_gathered, cb_partial_sumsq>(EltwiseShape::tiles(1));
            for (uint32_t k = 1; k < num_partials; ++k) {
                ckl::add<cb_partials_gathered, cb_partial_sumsq, cb_partial_sumsq>(EltwiseShape::tiles(1));
            }
        }

        // ---------- FINALIZE: rsqrt(sum * inv_W + eps) ----------
        ckl::eltwise_chain(
            EltwiseShape::tiles(1),
            CopyTile<cb_partial_sumsq, Dst::D0, InputLifecycle::Streaming>{},
            ckl::MulUnary<>{inv_W_bits},
            ckl::AddUnary<>{eps_bits},
            ckl::Rsqrt<>{},
            PackTile<cb_recip_rms, OutputLifecycle::Streaming>{});

        // ---------- PASS 2: normalize ----------
        if constexpr (has_gamma) {
            // x * recip (Col bcast) -> cb_normalized   (pops resident, pops recip at end)
            ckl::mul<
                cb_input_resident,
                cb_recip_rms,
                cb_normalized,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::Bulk>(EltwiseShape::tiles(Wt));

            // normalized * gamma (Row bcast) -> cb_output  (gamma held, indexed by Block)
            ckl::mul<
                cb_normalized,
                cb_gamma,
                cb_output,
                BroadcastDim::Row,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Block>(EltwiseShape::tiles(Wt));
        } else {
            ckl::mul<
                cb_input_resident,
                cb_recip_rms,
                cb_output,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::Bulk>(EltwiseShape::tiles(Wt));
        }
    }
}
