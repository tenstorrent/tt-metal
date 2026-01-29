// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/madd/device/madd_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

MAddOperation::spec_return_value_t MAddOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& a = tensor_args.a;

    const ttnn::Shape output_shape = a.logical_shape();
    const tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;
    const tt::tt_metal::DataType output_data_type = a.dtype();

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(output_data_type, tt::tt_metal::PageConfig(output_layout), args.output_mem_config));
}

MAddOperation::tensor_return_value_t MAddOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.a.device());
}

static void validate_operand(const Tensor& x) {
    TT_FATAL(x.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(x.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(x.layout() == tt::tt_metal::Layout::TILE, "Operands must be tiled!");

    const bool is_sharded = x.memory_config().is_sharded();
    if (!is_sharded) {
        TT_FATAL(
            x.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Only interleaved memory layout is supported for non-sharded tiled input");
    }

    TT_FATAL(x.padded_shape() == x.logical_shape(), "Only tile aligned tile input is currently supported");
}

void MAddOperation::validate_on_program_cache_miss(
    [[maybe_unused]] const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& c = tensor_args.c;

    // Check if inputs are sharded - all must be sharded or all must be interleaved
    const bool a_sharded = a.memory_config().is_sharded();
    const bool b_sharded = b.memory_config().is_sharded();
    const bool c_sharded = c.memory_config().is_sharded();

    TT_FATAL(
        (a_sharded == b_sharded) && (b_sharded == c_sharded),
        "All inputs must have the same memory layout (all sharded or all interleaved). "
        "Got: a={}, b={}, c={}",
        a_sharded ? "sharded" : "interleaved",
        b_sharded ? "sharded" : "interleaved",
        c_sharded ? "sharded" : "interleaved");

    validate_operand(a);
    validate_operand(b);
    validate_operand(c);

    // For sharded tensors, validate identical shard specs
    if (a_sharded) {
        const auto& shard_spec_a = a.shard_spec().value();
        const auto& shard_spec_b = b.shard_spec().value();
        const auto& shard_spec_c = c.shard_spec().value();

        TT_FATAL(shard_spec_a == shard_spec_b, "Shard spec mismatch: A and B must have identical sharding");
        TT_FATAL(shard_spec_b == shard_spec_c, "Shard spec mismatch: B and C must have identical sharding");

        // Validate shard shape is tile-aligned
        TT_FATAL(
            (shard_spec_a.shape[0] % tt::constants::TILE_HEIGHT == 0) &&
                (shard_spec_a.shape[1] % tt::constants::TILE_WIDTH == 0),
            "Shard shape must be tile-aligned. Got: {}x{}",
            shard_spec_a.shape[0],
            shard_spec_a.shape[1]);
    }

    TT_FATAL(
        a.logical_shape() == b.logical_shape(),
        "Matrix shape mismatch: A shape {} must equal B shape {}",
        a.logical_shape(),
        b.logical_shape());
    TT_FATAL(
        b.logical_shape() == c.logical_shape(),
        "Matrix shape mismatch: B shape {} must equal C shape {}",
        b.logical_shape(),
        c.logical_shape());
}

void MAddOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

MAddOperation::program_factory_t MAddOperation::select_program_factory(
    [[maybe_unused]] const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.a.memory_config().is_sharded()) {
        return MAddProgramFactorySharded{};
    }
    return MAddProgramFactory{};
}

ttnn::Tensor madd(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& c,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    return ttnn::device_operation::launch<MAddOperation>(
        MAddParams{.output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config},
        MAddArgs{.a = a, .b = b, .c = c});
}
}  // namespace ttnn::prim
