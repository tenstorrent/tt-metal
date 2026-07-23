// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void validate_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(tensor.buffer()->buffer_type() == BufferType::DRAM, "{} must be in DRAM", name);
    TT_FATAL(tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "{} must be interleaved", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE_LAYOUT", name);
    TT_FATAL(tensor.dtype() == DataType::FLOAT32, "{} must be FLOAT32, got {}", name, tensor.dtype());
}

void validate_shape(const Tensor& tensor, std::initializer_list<uint32_t> expected, const char* name) {
    const auto shape = tensor.logical_shape();
    TT_FATAL(shape.rank() == expected.size(), "{} rank {} != {}", name, shape.rank(), expected.size());
    size_t index = 0;
    for (const uint32_t dimension : expected) {
        TT_FATAL(
            static_cast<uint32_t>(shape[index]) == dimension,
            "{} dim[{}] {} != {}",
            name,
            index,
            shape[index],
            dimension);
        ++index;
    }
}

}  // namespace

void KDARecurrentDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& inputs) {
    validate_tensor(inputs.q_scaled, "q_scaled");
    validate_tensor(inputs.k_unit, "k_unit");
    validate_tensor(inputs.v, "v");
    validate_tensor(inputs.decay, "decay");
    validate_tensor(inputs.beta, "beta");
    validate_tensor(inputs.state, "state");

    const uint32_t heads = attributes.num_heads;
    const uint32_t key_dim = attributes.key_dim;
    const uint32_t value_dim = attributes.value_dim;
    TT_FATAL(heads > 0, "num_heads must be positive");
    TT_FATAL(
        key_dim % tt::constants::TILE_WIDTH == 0,
        "key_dim {} must be divisible by {}",
        key_dim,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        value_dim % tt::constants::TILE_WIDTH == 0,
        "value_dim {} must be divisible by {}",
        value_dim,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        attributes.output_memory_config.buffer_type() == BufferType::DRAM &&
            attributes.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "output memory config must be interleaved DRAM");

    validate_shape(inputs.q_scaled, {heads, 1, key_dim}, "q_scaled");
    validate_shape(inputs.k_unit, {heads, 1, key_dim}, "k_unit");
    validate_shape(inputs.v, {heads, 1, value_dim}, "v");
    validate_shape(inputs.decay, {heads, key_dim, 1}, "decay");
    validate_shape(inputs.beta, {heads, 1, 1}, "beta");
    validate_shape(inputs.state, {heads, key_dim, value_dim}, "state");
}

void KDARecurrentDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attributes, inputs);
}

KDARecurrentDeviceOperation::spec_return_value_t KDARecurrentDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, [[maybe_unused]] const tensor_args_t& inputs) {
    const auto layout = [&](const Shape& shape) {
        return TensorSpec(
            shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attributes.output_memory_config));
    };
    return {
        layout(Shape({attributes.num_heads, 1, attributes.value_dim})),
        layout(Shape({attributes.num_heads, attributes.key_dim, attributes.value_dim})),
    };
}

KDARecurrentDeviceOperation::tensor_return_value_t KDARecurrentDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& inputs) {
    const auto specs = compute_output_specs(attributes, inputs);
    return {
        create_device_tensor(specs[0], inputs.q_scaled.device()),
        create_device_tensor(specs[1], inputs.q_scaled.device()),
    };
}

ttsl::hash::hash_t KDARecurrentDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& inputs) {
    return operation::hash_operation<KDARecurrentDeviceOperation>(
        attributes.num_heads,
        attributes.key_dim,
        attributes.value_dim,
        attributes.output_memory_config,
        attributes.compute_kernel_config,
        inputs.q_scaled,
        inputs.k_unit,
        inputs.v,
        inputs.decay,
        inputs.beta,
        inputs.state);
}

std::vector<Tensor> kda_recurrent_step(
    const Tensor& q_scaled,
    const Tensor& k_unit,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    const MemoryConfig& output_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Operation = KDARecurrentDeviceOperation;
    const auto q_shape = q_scaled.logical_shape();
    const auto v_shape = v.logical_shape();
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .num_heads = static_cast<uint32_t>(q_shape[0]),
            .key_dim = static_cast<uint32_t>(q_shape[2]),
            .value_dim = static_cast<uint32_t>(v_shape[2]),
            .output_memory_config = output_memory_config,
            .compute_kernel_config = compute_kernel_config,
        },
        Operation::tensor_args_t{
            .q_scaled = q_scaled,
            .k_unit = k_unit,
            .v = v,
            .decay = decay,
            .beta = beta,
            .state = state,
        });
}

}  // namespace ttnn::prim
