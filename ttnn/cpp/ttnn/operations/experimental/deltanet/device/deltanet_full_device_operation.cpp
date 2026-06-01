// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_full_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deltanet {

DeltaNetDecodeFullDeviceOperation::program_factory_t DeltaNetDecodeFullDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*inputs*/) {
    return DeltaNetDecodeFullProgramFactory{};
}

void DeltaNetDecodeFullDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    TT_FATAL(
        inputs.recurrent_state.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: recurrent_state must be on device");
    TT_FATAL(
        inputs.qkv_proj.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: qkv_proj must be on device");
    TT_FATAL(
        inputs.z_proj.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: z_proj must be on device");
    TT_FATAL(
        inputs.conv_state.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: conv_state must be on device");
    TT_FATAL(
        inputs.conv1d_weight.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: conv1d_weight must be on device");
    TT_FATAL(
        inputs.conv_state.layout() == Layout::TILE,
        "DeltaNet decode full: conv_state must be TILE layout");
    TT_FATAL(
        inputs.recurrent_state.layout() == Layout::TILE,
        "DeltaNet decode full: recurrent_state must be TILE layout");
    TT_FATAL(
        attrs.k_head_dim % 32 == 0 && attrs.v_head_dim % 32 == 0,
        "DeltaNet decode full: head dims must be multiples of 32");
}

void DeltaNetDecodeFullDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attrs, inputs);
}

DeltaNetDecodeFullDeviceOperation::spec_return_value_t DeltaNetDecodeFullDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto mem_config = attrs.output_memory_config;

    auto dtype = inputs.qkv_proj.dtype();

    // output: [1, 1, 1, num_heads * v_dim] — flat gated output ready for o_proj
    auto output_shape = Shape({1, 1, 1, attrs.num_heads * attrs.v_head_dim});
    auto output_spec = TensorSpec(
        output_shape,
        TensorLayout(dtype, Layout::TILE, mem_config));

    // new_state: same shape as recurrent_state [1, H, Dk, Dv], bf16 (same as compute)
    auto state_spec = TensorSpec(
        inputs.recurrent_state.logical_shape(),
        TensorLayout(dtype, Layout::TILE, mem_config));

    // new_conv_state: same shape as input conv_state [1, 1, conv_dim, 32]
    auto conv_state_spec = TensorSpec(
        inputs.conv_state.logical_shape(),
        TensorLayout(dtype, Layout::TILE, mem_config));

    return {output_spec, state_spec, conv_state_spec};
}

DeltaNetDecodeFullDeviceOperation::tensor_return_value_t DeltaNetDecodeFullDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs,
    const tensor_args_t& inputs) {
    auto* device = inputs.recurrent_state.device();
    auto output_specs = compute_output_specs(attrs, inputs);
    return {
        create_device_tensor(output_specs[0], device),
        create_device_tensor(output_specs[1], device),
        create_device_tensor(output_specs[2], device),
    };
}

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& qkv_proj,
    const Tensor& z_proj,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& conv_state,
    const Tensor& recurrent_state,
    const Tensor& conv1d_weight,
    const Tensor& a_log,
    const Tensor& dt_bias,
    const Tensor& norm_weight,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t conv_dim,
    uint32_t conv_kernel_size,
    uint32_t head_expand_ratio,
    const std::optional<MemoryConfig>& output_memory_config) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetDecodeFullDeviceOperation;

    auto mem_config = output_memory_config.value_or(recurrent_state.memory_config());

    auto operation_attributes = Op::operation_attributes_t{
        .num_heads = num_heads,
        .num_k_heads = num_k_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .conv_dim = conv_dim,
        .conv_kernel_size = conv_kernel_size,
        .head_expand_ratio = head_expand_ratio,
        .output_memory_config = mem_config,
    };

    auto tensor_args = Op::tensor_args_t{
        .qkv_proj = qkv_proj,
        .z_proj = z_proj,
        .b_proj = b_proj,
        .a_proj = a_proj,
        .conv_state = conv_state,
        .recurrent_state = recurrent_state,
        .conv1d_weight = conv1d_weight,
        .a_log = a_log,
        .dt_bias = dt_bias,
        .norm_weight = norm_weight,
    };

    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
