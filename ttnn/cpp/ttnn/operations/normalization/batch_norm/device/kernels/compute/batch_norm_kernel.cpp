// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

#include "api/dataflow/circular_buffer.h"

// batchnorm_bcast_tiles: computes one block of (input - batch_mean) / sqrt(batch_var + eps),
// optionally scaled by weight and offset by bias. Layout:
//
//   Stage 1 (one tile):  cb_den = rsqrt(cb_batch_var + cb_eps)
//   Stage 2..4 fused:    cb_output_0 = ((cb_other - cb_bcast) * cb_den [* cb_weight]) [+ cb_bias]
//
// The fused stage uses BinaryFpu followed by DestReuseBinary elements so the running result
// stays in DEST[0] across the entire chain — no intermediate CB writes/reads (the original
// kernel staged through cb_tmp_1 to bridge separate tile_regs windows; the chain owns the
// dst-sync window so the bridge is unnecessary).
//
// CB lifecycles:
//   cb_batch_var      InputLifecycle::Bulk on Scalar          chain waits 1 / pops 1 within stage 1
//   cb_eps            InputLifecycle::CallerManaged on Scalar held by kernel_main across the whole kernel
//   cb_other          InputLifecycle::Streaming on Scalar     per-tile wait/pop, reads CB front each iter
//                                             (Scalar idx = 0; pop advances the front so each
//                                             iter consumes the next producer-pushed tile —
//                                             matches the original `sub_tiles(cb_other, _, 0, _, _)`)
//   cb_bcast, cb_den  InputLifecycle::CallerManaged on Scalar caller waits before the chain, pops after
//   cb_weight,
//   cb_bias           InputLifecycle::CallerManaged on Scalar same — held by this function call only
template <bool WeightHas, bool BiasHas>
ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_output_0) {
    // Stage 1: cb_den = 1 / sqrt(cb_batch_var + cb_eps), one tile.
    compute_kernel_lib::eltwise_chain(
        1,
        compute_kernel_lib::BinaryFpu<
            cb_batch_var,
            cb_eps,
            compute_kernel_lib::BinaryFpuOp::Add,
            compute_kernel_lib::BroadcastDim::None,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::Rsqrt<>{},
        compute_kernel_lib::
            PackTile<cb_den, compute_kernel_lib::Dst::D0, compute_kernel_lib::OutputLifecycle::Streaming>{});

    const uint32_t inner_count = freq - tile_start;

    // Wait the operands that live across the entire stage-2+ chain (InputLifecycle::CallerManaged on the chain
    // means the chain emits no wait/pop edges on these — the caller owns them).
    cb_wait_front(cb_bcast, 1);
    cb_wait_front(cb_den, 1);
    if constexpr (WeightHas) {
        cb_wait_front(cb_weight, 1);
    }
    if constexpr (BiasHas) {
        cb_wait_front(cb_bias, 1);
    }

    // Reusable chain pieces. Sub walks cb_other producer-streamed (Scalar idx + InputLifecycle::Streaming
    // wait/pop drains the producer one tile per iter); the three DestReuse multiplies/adds
    // are bcast operands held across the whole chain (InputLifecycle::CallerManaged on Scalar). The
    // weight / bias multiplies are OptionalChainElement-gated on the template bools so
    // they collapse to no-op tag wrappers when the caller didn't pass those tensors.
    constexpr auto sub_op = compute_kernel_lib::BinaryFpu<
        cb_other,
        cb_bcast,
        compute_kernel_lib::BinaryFpuOp::Sub,
        compute_kernel_lib::BroadcastDim::None,
        compute_kernel_lib::BinaryDataFormatReconfig::Input,
        compute_kernel_lib::InputLifecycle::Streaming,
        compute_kernel_lib::InputLifecycle::CallerManaged,
        compute_kernel_lib::OperandKind::Scalar,
        compute_kernel_lib::Dst::D0,
        compute_kernel_lib::OperandKind::Scalar>{};
    constexpr auto mul_den = compute_kernel_lib::DestReuseBinary<
        cb_den,
        compute_kernel_lib::BinaryFpuOp::Mul,
        compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
        compute_kernel_lib::Dst::D0,
        compute_kernel_lib::Dst::D0,
        compute_kernel_lib::DestReuseReconfig::Input,
        compute_kernel_lib::InputLifecycle::CallerManaged,
        compute_kernel_lib::OperandKind::Scalar>{};
    constexpr auto mul_weight = compute_kernel_lib::OptionalChainElement<
        WeightHas,
        compute_kernel_lib::DestReuseBinary<
            cb_weight,
            compute_kernel_lib::BinaryFpuOp::Mul,
            compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::DestReuseReconfig::Input,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OperandKind::Scalar>>{};
    constexpr auto add_bias = compute_kernel_lib::OptionalChainElement<
        BiasHas,
        compute_kernel_lib::DestReuseBinary<
            cb_bias,
            compute_kernel_lib::BinaryFpuOp::Add,
            compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::DestReuseReconfig::Input,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OperandKind::Scalar>>{};
    constexpr auto pack_out = compute_kernel_lib::
        PackTile<cb_output_0, compute_kernel_lib::Dst::D0, compute_kernel_lib::OutputLifecycle::Streaming>{};

    // Stage 2..4 fused. Single chain — DEST[0] threaded through Sub → Mul(den) →
    // [Mul(weight)] → [Add(bias)] → Pack. Optional weight / bias gates handle the four
    // (WeightHas, BiasHas) cases without a four-way constexpr-if.
    compute_kernel_lib::eltwise_chain(inner_count, sub_op, mul_den, mul_weight, add_bias, pack_out);

    cb_pop_front(cb_bcast, 1);
    cb_pop_front(cb_den, 1);
    if constexpr (WeightHas) {
        cb_pop_front(cb_weight, 1);
    }
    if constexpr (BiasHas) {
        cb_pop_front(cb_bias, 1);
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 = get_compile_time_arg_val(4);    // output
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);   // batch_var
    constexpr auto cb_eps = get_compile_time_arg_val(6);         // eps
    constexpr auto cb_den = get_compile_time_arg_val(7);         // 1/sqrt(batch_var + eps)
    constexpr auto cb_weight = get_compile_time_arg_val(8);      // weight tensor
    // get_compile_time_arg_val(9) used to be cb_tmp_1 — no longer referenced (the fused chain
    // keeps the running result in DEST instead of staging through cb_tmp_1). CT-arg slot kept
    // for ABI compatibility with the program factory.
    constexpr auto cb_bias = get_compile_time_arg_val(10);  // bias tensor

    binary_op_init_common(cb_input, cb_batch_mean, cb_output_0);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    cb_wait_front(cb_eps, 1);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value>(
            cb_batch_mean,
            cb_input,
            tile_freq,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value>(
            cb_batch_mean,
            cb_input,
            remaining_iterations,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0);
    }

    cb_pop_front(cb_eps, 1);
}
