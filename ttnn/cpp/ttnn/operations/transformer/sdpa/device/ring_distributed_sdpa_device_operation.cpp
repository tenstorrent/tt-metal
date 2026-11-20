// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_ring_distributed.hpp"
#include "ring_distributed_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "sdpa_perf_model.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
void RingDistributedSdpaDeviceOperation::validate_on_program_cache_miss(
    const RingDistributedSDPAParams& operation_attributes, const RingDistributedSDPAInputs& tensor_args) {
    // resolve_ring_distributed_recipe checks the ring arguments before the program-cache lookup; recipe_program checks
    // shapes, types and layouts when it builds the program.
    TT_FATAL(
        operation_attributes.precision.has_value(),
        "Ring-distributed SDPA: the precision recipe is resolved before dispatch");
    (void)tensor_args;
}

tt::tt_metal::TensorSpec RingDistributedSdpaDeviceOperation::compute_output_specs(
    const RingDistributedSDPAParams& operation_attributes, const RingDistributedSDPAInputs& tensor_args) {
    return ttnn::operations::transformer::sdpa::detail::ring_distributed_recipe_output_spec(
        operation_attributes, tensor_args);
}

Tensor RingDistributedSdpaDeviceOperation::create_output_tensors(
    const RingDistributedSDPAParams& operation_attributes, const RingDistributedSDPAInputs& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<RingDistributedSdpaDeviceOperation::tensor_return_value_t>
RingDistributedSdpaDeviceOperation::create_op_performance_model(
    const RingDistributedSDPAParams& args, const RingDistributedSDPAInputs& tensor_args, Tensor& output_tensor) {
    Tensors input_tensors = {tensor_args.q, tensor_args.k, tensor_args.v};

    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    // Performance model only supports Wormhole B0 and Blackhole architectures
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "SDPA perf model does not support tt::arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output_tensor, 0);
    }

    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    const auto& v_shape = tensor_args.v.logical_shape();
    const auto& output_shape = output_tensor.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    tt::tt_metal::MathFidelity fidelity = ttnn::operations::transformer::sdpa::detail::recipe_math_fidelity(args.precision, tensor_args.k.dtype());

    // In chunked/paged-KV mode, k_shape[2] is the page block size, not the full KV sequence length.
    // Use chunk_start_idx + q_shape[2] as the effective Sk (similar to SDPAOperation).
    const bool is_chunked = args.chunk_start_idx.has_value();
    const uint32_t Sk = is_chunked ? (q_shape[2] + args.chunk_start_idx.value()) : k_shape[2];

    // For ring distributed SDPA, use local output sequence length (Sq) and full K sequence length (Sk)
    // Ring distributed SDPA is always causal
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        q_shape[0],       // batch
        q_shape[1],       // num_heads_q
        output_shape[2],  // Sq (local output seq len)
        Sk,               // Sk (effective K seq len, adjusted for chunked mode)
        q_shape[3],       // DH
        v_shape[3],       // DV
        true,             // is_causal (always true for ring distributed)
        fidelity,
        grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output_tensor, ideal_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {

Tensor ring_distributed_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const std::optional<ttnn::Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx,
    std::optional<ttnn::operations::transformer::SDPAPrecision> precision) {
    using OperationType = ttnn::prim::RingDistributedSdpaDeviceOperation;

    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }
    auto operation_attributes = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_id = ring_id,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .chunk_start_idx = chunk_start_idx,
        .precision = precision.value_or(ttnn::operations::transformer::SDPAPrecision::STANDARD),
    };

    auto tensor_args = OperationType::tensor_args_t{
        .q = input_tensor_q,
        .k = input_tensor_k,
        .v = input_tensor_v,
        .page_table = page_table,
    };
    ttnn::operations::transformer::sdpa::detail::resolve_ring_distributed_recipe(operation_attributes, tensor_args);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
