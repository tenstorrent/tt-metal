// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_scaled_fp8_kv_cache_device_operation.hpp"

#include <limits>

#include <tt_stl/small_vector.hpp>

#include <tt-metalium/hal.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/pack_scaled_fp8_kv_cache.hpp"

namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache {

namespace packed = ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache;

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& config) {
    return config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_input(const Tensor& tensor, const char* name, tt::tt_metal::DataType dtype, uint32_t width) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
    TT_FATAL(tensor.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(!tensor.logical_shape().empty(), "{} must have at least one dimension", name);
    TT_FATAL(
        tensor.logical_shape()[-1] == width, "{} last dim must be {}, got {}", name, width, tensor.logical_shape()[-1]);
}

}  // namespace

PackScaledFp8KvCacheDeviceOperation::program_factory_t PackScaledFp8KvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PackScaledFp8KvCacheProgramFactory{};
}

void PackScaledFp8KvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_input(
        args.latent, "pack_scaled_fp8_kv_cache: latent", tt::tt_metal::DataType::FP8_E4M3, packed::LATENT_WIDTH);
    validate_input(
        args.scales, "pack_scaled_fp8_kv_cache: scales", tt::tt_metal::DataType::FLOAT32, packed::SCALE_WIDTH);
    validate_input(args.rope, "pack_scaled_fp8_kv_cache: rope", tt::tt_metal::DataType::BFLOAT16, packed::ROPE_WIDTH);
    TT_FATAL(
        is_dram_interleaved(attrs.output_memory_config), "pack_scaled_fp8_kv_cache: output must be DRAM interleaved");
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "pack_scaled_fp8_kv_cache requires Blackhole");
    TT_FATAL(
        args.latent.device() == args.scales.device() && args.latent.device() == args.rope.device(),
        "all inputs must be on the same device");

    const auto& shape = args.latent.logical_shape();
    TT_FATAL(
        shape.size() == args.scales.logical_shape().size() && shape.size() == args.rope.logical_shape().size(),
        "all inputs must have the same rank");
    uint64_t rows = 1;
    for (size_t dim = 0; dim + 1 < shape.size(); ++dim) {
        TT_FATAL(
            shape[dim] == args.scales.logical_shape()[dim] && shape[dim] == args.rope.logical_shape()[dim],
            "all inputs must have identical leading shapes");
        rows *= static_cast<uint64_t>(shape[dim]);
        TT_FATAL(rows <= std::numeric_limits<uint32_t>::max(), "folded row count exceeds uint32_t");
    }
    TT_FATAL(rows > 0, "row count must be positive");
}

void PackScaledFp8KvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

PackScaledFp8KvCacheDeviceOperation::spec_return_value_t PackScaledFp8KvCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& input_shape = args.latent.logical_shape();
    ttsl::SmallVector<uint32_t> dims;
    dims.reserve(input_shape.size());
    for (size_t dim = 0; dim + 1 < input_shape.size(); ++dim) {
        dims.push_back(static_cast<uint32_t>(input_shape[dim]));
    }
    dims.push_back(packed::PACKED_ROW_BYTES);
    return TensorSpec(
        ttnn::Shape(dims),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FP8_E4M3,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_memory_config));
}

PackScaledFp8KvCacheDeviceOperation::tensor_return_value_t PackScaledFp8KvCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.latent.device());
}

ttsl::hash::hash_t PackScaledFp8KvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return tt::tt_metal::operation::hash_operation<PackScaledFp8KvCacheDeviceOperation>(
        attrs,
        args.latent.memory_config(),
        args.scales.memory_config(),
        args.rope.memory_config(),
        args.latent.logical_shape());
}

}  // namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache

namespace ttnn::prim {

ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const Tensor& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using Operation = ttnn::experimental::prim::pack_scaled_fp8_kv_cache::PackScaledFp8KvCacheDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{.output_memory_config = output_memory_config},
        Operation::tensor_args_t{.latent = latent, .scales = scales, .rope = rope});
}

}  // namespace ttnn::prim
