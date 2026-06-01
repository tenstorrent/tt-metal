// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sinkhorn_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

void validate_tensors(const SinkhornParams& attributes, const SinkhornInputs& tensor_args) {
    const auto& comb_w = tensor_args.comb_w;
    const auto& comb_bias = tensor_args.comb_bias;

    TT_FATAL(comb_w.storage_type() == StorageType::DEVICE, "fused_hyperconnection_sinkhorn: comb_w must be on device");
    TT_FATAL(
        comb_bias.storage_type() == StorageType::DEVICE, "fused_hyperconnection_sinkhorn: comb_bias must be on device");

    TT_FATAL(comb_w.layout() == Layout::TILE, "fused_hyperconnection_sinkhorn: comb_w must be TILE layout");
    TT_FATAL(comb_bias.layout() == Layout::TILE, "fused_hyperconnection_sinkhorn: comb_bias must be TILE layout");

    TT_FATAL(comb_w.dtype() == DataType::BFLOAT16, "fused_hyperconnection_sinkhorn: comb_w must be BFLOAT16");
    TT_FATAL(comb_bias.dtype() == DataType::BFLOAT16, "fused_hyperconnection_sinkhorn: comb_bias must be BFLOAT16");

    const auto& shape = comb_w.logical_shape();
    TT_FATAL(
        shape.rank() == 4,
        "fused_hyperconnection_sinkhorn: comb_w must be rank-4 [1,1,H,H], got rank {}",
        shape.rank());
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "fused_hyperconnection_sinkhorn: comb_w must be [1,1,H,H] (decode, T==1)");
    TT_FATAL(
        shape[2] == attributes.num_streams && shape[3] == attributes.num_streams,
        "fused_hyperconnection_sinkhorn: comb_w must be [1,1,H,H] with H={}, got [{},{}]",
        attributes.num_streams,
        shape[2],
        shape[3]);
    TT_FATAL(
        attributes.num_streams <= 32,
        "fused_hyperconnection_sinkhorn: only H<=32 (single tile) is supported, got H={}",
        attributes.num_streams);
    TT_FATAL(
        comb_w.logical_shape() == comb_bias.logical_shape(),
        "fused_hyperconnection_sinkhorn: comb_w and comb_bias must have the same shape");
    TT_FATAL(attributes.sinkhorn_iters >= 1, "fused_hyperconnection_sinkhorn: sinkhorn_iters must be >= 1");
}

}  // namespace

void SinkhornDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

void SinkhornDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

SinkhornDeviceOperation::spec_return_value_t SinkhornDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& comb_w = tensor_args.comb_w;
    const auto output_layout = tt::tt_metal::TensorLayout(
        comb_w.dtype(), tt::tt_metal::PageConfig(comb_w.layout()), operation_attributes.output_mem_config);
    return TensorSpec(comb_w.logical_shape(), output_layout);
}

SinkhornDeviceOperation::tensor_return_value_t SinkhornDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.comb_w.device());
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

Tensor fused_hyperconnection_sinkhorn(
    const Tensor& comb_w,
    const Tensor& comb_bias,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::hyperconnection::SinkhornDeviceOperation;
    const MemoryConfig output_mem_config = memory_config.value_or(comb_w.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = num_streams,
        .sinkhorn_iters = sinkhorn_iters,
        .comb_scale = comb_scale,
        .eps = eps,
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .comb_w = comb_w,
        .comb_bias = comb_bias,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
