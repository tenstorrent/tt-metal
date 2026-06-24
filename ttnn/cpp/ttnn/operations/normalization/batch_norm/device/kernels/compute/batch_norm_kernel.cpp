// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"     // BinaryFpu, DestReuseBinary, PackTile, eltwise_chain
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"      // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

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
//   cb_bcast, cb_den  InputLifecycle::Bulk on Scalar           chain waits 1 / pops 1 per call
//   cb_weight,
//   cb_bias           InputLifecycle::Bulk on Scalar (Optional) same; gated off => chain emits no CB edges
// CB ids are non-type template params (the chain elements take them as template args, which
// requires constant expressions); only the per-call tile counts stay runtime.
template <
    bool WeightHas,
    bool BiasHas,
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_output_0>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    // Stage 1: cb_den = 1 / sqrt(cb_batch_var + cb_eps), one tile.
    ckl::eltwise_chain(
        ckl::EltwiseShape::single(),
        ckl::BinaryFpu<
            cb_batch_var,
            cb_eps,
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::Bulk,
            ckl::InputLifecycle::CallerManaged>{},
        ckl::Rsqrt<>{},
        ckl::PackTile<cb_den>{});

    const uint32_t inner_count = freq - tile_start;

    // Reusable chain pieces. Sub walks cb_other producer-streamed (Scalar idx + InputLifecycle::Streaming
    // wait/pop drains the producer one tile per iter); the three DestReuse multiplies/adds are
    // single-tile bcast operands -> InputLifecycle::Bulk + Scalar (the chain owns one wait(1)/pop(1)
    // per call, mirroring batch_norm_sfpu_kernel.cpp). The weight / bias multiplies are
    // OptionalChainElement-gated on the template bools so they collapse to no-op tag wrappers (no CB
    // edges — wait/pop suppressed) when the caller didn't pass those tensors.
    constexpr auto sub_op = ckl::BinaryFpu<
        cb_other,
        cb_bcast,
        ckl::BinaryFpuOp::Sub,
        ckl::BroadcastDim::None,
        ckl::InputLifecycle::Streaming,
        ckl::InputLifecycle::Bulk>{};  // cb_bcast: held scalar, chain owns wait(1)/pop(1) per call
    constexpr auto mul_den = ckl::
        DestReuseBinary<cb_den, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA, ckl::InputLifecycle::Bulk>{};
    constexpr auto mul_weight = ckl::OptionalChainElement<
        WeightHas,
        ckl::DestReuseBinary<
            cb_weight,
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA,
            ckl::InputLifecycle::Bulk>>{};
    constexpr auto add_bias = ckl::OptionalChainElement<
        BiasHas,
        ckl::DestReuseBinary<
            cb_bias,
            ckl::BinaryFpuOp::Add,
            ckl::DestReuseType::DEST_TO_SRCA,
            ckl::InputLifecycle::Bulk>>{};
    constexpr auto pack_out = ckl::PackTile<cb_output_0>{};

    // Stage 2..4 fused. Single chain — DEST[0] threaded through Sub → Mul(den) →
    // [Mul(weight)] → [Add(bias)] → Pack. Optional weight / bias gates handle the four
    // (WeightHas, BiasHas) cases without a four-way constexpr-if.
    ckl::eltwise_chain(ckl::EltwiseShape::tiles(inner_count), sub_op, mul_den, mul_weight, add_bias, pack_out);
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
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0>(remaining_iterations, tile_start);
    }

    cb_pop_front(cb_eps, 1);
}
