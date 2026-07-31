// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async.hpp"

#include "device/all_gather_regime_a_matmul_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

namespace {

// Validate the K-sharded contract and return the TP group size implied by the shapes.
//
// in0 is [.., M, K/TP] (this device's K shard); in1 is [.., K, N] (full K, device-local). The TP group
// size is therefore in1.K / in0.K -- it is derived rather than passed so the two operands can never
// disagree about how the activation was sharded.
uint32_t validate_and_infer_tp(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore) {
    const auto& in0_shape = input_tensor.logical_shape();
    const auto& in1_shape = weight_tensor.logical_shape();
    TT_FATAL(in0_shape.rank() >= 2, "in0 must be rank >= 2, got rank {}", in0_shape.rank());
    TT_FATAL(in1_shape.rank() >= 2, "in1 must be rank >= 2, got rank {}", in1_shape.rank());

    const uint32_t k_local = in0_shape[-1];
    const uint32_t k_global = in1_shape[-2];
    TT_FATAL(k_local > 0, "in0 K shard must be non-empty");
    TT_FATAL(
        k_global % k_local == 0,
        "in1 K ({}) must be a whole multiple of the in0 K shard ({}); in0 is expected to be [M, K/TP] and "
        "in1 the full [K, N]",
        k_global,
        k_local);

    const uint32_t tp = k_global / k_local;
    TT_FATAL(tp >= 2, "all_gather_regime_a_matmul_async needs a TP group of >= 2, inferred {}", tp);
    TT_FATAL(tp % 2 == 0, "the design spec requires an EVEN TP group (bidirectional schedule), inferred {}", tp);
    TT_FATAL(
        !multi_device_global_semaphore.empty(),
        "all_gather_regime_a_matmul_async requires at least one global semaphore for the all-gather");
    return tp;
}

// PHASE 0: materialise the full [M, K] activation with a standalone all-gather.
//
// This is deliberately NOT overlapped -- it is the correctness oracle and the unfused baseline. The fused
// implementation replaces this call with progressive, wave-scheduled ingress (see the design spec).
//
// CALLER CONTRACT for `persistent_output_buffer` and `multi_device_global_semaphore` (see
// models/tt_dit/parallel/manager.py, CCLManager):
//   1. Both must be DOUBLE-BUFFERED (ping-ponged) when the op is invoked repeatedly, so the resources
//      handed in are free of the previous invocation's still-in-flight CCL traffic.
//   2. ttnn.synchronize_device() must be called AFTER allocating them and BEFORE first use. Global
//      semaphores and buffers are created with per-device work; without the barrier a fast device can
//      launch the op and fire a cross-device atomic-inc at a peer that has not allocated/zeroed its copy
//      yet, and the increment is silently lost.
// Violating either produces intermittent, per-device PARTIAL corruption -- measured here as PCC
// 0.967-0.984 against a 0.999 target on a subset of devices, recovering on the next iteration. It looks
// like a program-cache bug but is not: the bad output does not correlate with the previous iteration's
// result (~0.001).
ttnn::Tensor gather_activation(
    const ttnn::Tensor& input_tensor,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::experimental::all_gather_async(
        input_tensor,
        persistent_output_buffer,
        /*dim=*/-1,
        multi_device_global_semaphore,
        /*num_links=*/num_links,
        /*memory_config=*/std::nullopt,
        topology,
        /*subdevice_id=*/std::nullopt,
        cluster_axis,
        /*use_optimal_ccl_for_llama=*/false,
        barrier_semaphore);
}

}  // namespace

ttnn::Tensor all_gather_regime_a_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis) {
    validate_and_infer_tp(input_tensor, weight_tensor, multi_device_global_semaphore);

    const ttnn::Tensor gathered = gather_activation(
        input_tensor,
        multi_device_global_semaphore,
        barrier_semaphore,
        persistent_output_buffer,
        num_links,
        topology,
        cluster_axis);

    auto outs = ttnn::prim::all_gather_regime_a_matmul_async(
        gathered,
        weight_tensor,
        config,
        bias_tensor,
        std::move(fused_activation),
        fused_ternary_scalar,
        fused_ternary_input_a,
        fused_ternary_input_b,
        1);  // chunks
    TT_FATAL(outs.size() == 1, "all_gather_regime_a_matmul_async expected a single output, got {}", outs.size());
    return outs[0];
}

std::vector<ttnn::Tensor> all_gather_regime_a_matmul_async_split(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    int32_t chunks,
    int32_t dim,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis) {
    TT_FATAL(chunks >= 1, "all_gather_regime_a_matmul_async_split requires chunks >= 1, got {}", chunks);
    // `dim` is kept in the wrapper signature for minimal_matmul API compatibility, validated here, and NOT
    // forwarded to the device op (only -1 is supported; the device op works on `chunks` alone).
    TT_FATAL(dim == -1, "all_gather_regime_a_matmul_async_split only supports dim=-1, got {}", dim);
    validate_and_infer_tp(input_tensor, weight_tensor, multi_device_global_semaphore);

    const ttnn::Tensor gathered = gather_activation(
        input_tensor,
        multi_device_global_semaphore,
        barrier_semaphore,
        persistent_output_buffer,
        num_links,
        topology,
        cluster_axis);

    return ttnn::prim::all_gather_regime_a_matmul_async(
        gathered,
        weight_tensor,
        config,
        bias_tensor,
        std::move(fused_activation),
        fused_ternary_scalar,
        fused_ternary_input_a,
        fused_ternary_input_b,
        chunks);
}

}  // namespace ttnn::experimental
