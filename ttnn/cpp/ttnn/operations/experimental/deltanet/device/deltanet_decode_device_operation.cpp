// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deltanet {

DeltaNetDecodeDeviceOperation::program_factory_t DeltaNetDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*inputs*/) {
    return DeltaNetDecodeProgramFactory{};
}

void DeltaNetDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    TT_FATAL(
        inputs.state.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: state must be on device");
    TT_FATAL(
        inputs.query.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: query must be on device");
    TT_FATAL(
        inputs.key.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: key must be on device");
    TT_FATAL(
        inputs.value.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: value must be on device");
    TT_FATAL(
        inputs.decay.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: decay must be on device");
    TT_FATAL(
        inputs.beta.storage_type() == StorageType::DEVICE,
        "DeltaNet decode: beta must be on device");

    TT_FATAL(
        inputs.state.layout() == Layout::TILE,
        "DeltaNet decode: all inputs must be TILE layout");
    TT_FATAL(
        attrs.k_head_dim % 32 == 0 && attrs.v_head_dim % 32 == 0,
        "DeltaNet decode: head dims must be multiples of 32");
}

void DeltaNetDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attrs, inputs);
}

DeltaNetDecodeDeviceOperation::spec_return_value_t DeltaNetDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto dtype = inputs.state.dtype();
    auto mem_config = attrs.output_memory_config;

    // output: [1, num_heads, 1, v_dim] — same shape as value
    auto output_spec = TensorSpec(
        inputs.value.logical_shape(),
        TensorLayout(dtype, Layout::TILE, mem_config));

    // new_state: same shape as input state [1, num_heads, k_dim, v_dim]
    auto state_spec = TensorSpec(
        inputs.state.logical_shape(),
        TensorLayout(dtype, Layout::TILE, mem_config));

    return {output_spec, state_spec};
}

DeltaNetDecodeDeviceOperation::tensor_return_value_t DeltaNetDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs,
    const tensor_args_t& inputs) {
    auto* device = inputs.state.device();
    auto output_specs = compute_output_specs(attrs, inputs);
    return {
        create_device_tensor(output_specs[0], device),
        create_device_tensor(output_specs[1], device),
    };
}

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    const std::optional<MemoryConfig>& output_memory_config) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetDecodeDeviceOperation;

    auto mem_config = output_memory_config.value_or(state.memory_config());

    auto operation_attributes = Op::operation_attributes_t{
        .num_heads = num_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .output_memory_config = mem_config,
    };

    auto tensor_args = Op::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .decay = decay,
        .beta = beta,
        .state = state,
    };

    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
