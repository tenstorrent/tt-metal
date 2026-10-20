// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_cache_device_operation.hpp"
#include "../update_cache.hpp"
#include "../validate_descriptor.hpp"

#include <algorithm>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::gemma4 {

using namespace tt::tt_metal;

void UpdateCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& args) {
    for (const auto* tensor : {&args.cache, &args.input}) {
        TT_FATAL(
            tensor->storage_type() == StorageType::DEVICE && tensor->buffer() != nullptr &&
                tensor->device() == args.input.device() && tensor->layout() == Layout::TILE,
            "Packed cache updates require allocated TILE tensors on one device");
        TT_FATAL(
            tensor->logical_shape().rank() == 4 && tensor->logical_shape() == tensor->padded_shape(),
            "Packed cache updates require four-dimensional, tile-aligned tensors");
    }
    TT_FATAL(args.cache.buffer()->buffer_type() == BufferType::DRAM, "Packed KV cache must reside in DRAM");
    TT_FATAL(
        args.input.dtype() == args.cache.dtype() && args.input.buffer() != args.cache.buffer(),
        "Packed cache updates require distinct input/cache buffers with matching dtypes");
    TT_FATAL(
        args.input.dtype() == DataType::BFLOAT16 || args.input.dtype() == DataType::BFLOAT8_B ||
            args.input.dtype() == DataType::BFLOAT4_B,
        "Packed cache updates support BF16, BFP8, and BFP4");
    const auto& input = args.input.logical_shape();
    const auto& cache = args.cache.logical_shape();
    TT_FATAL(
        input[0] == 1 && input[1] == cache[1] && input[3] == cache[3],
        "Packed input must have batch one and match cache head count and width");
    TT_FATAL(
        params.num_layers > 0 && params.layer_idx < params.num_layers && cache[0] % params.num_layers == 0,
        "Invalid cache layer fold");
    TT_FATAL(
        params.cluster_axis < args.input.device()->shape().dims(), "cluster_axis must name an axis of the input mesh");
    TT_FATAL(
        params.request_local_tokens > 0 && params.request_local_tokens % 32 == 0 &&
            params.request_local_tokens <= input[2],
        "request_local_tokens must be a positive multiple of 32 fitting the packed input");
    validate_request_descriptors(args.descriptors, args.input, params.request_index);
}

void UpdateCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& params, const tensor_args_t& args) {
    validate_on_program_cache_miss(params, args);
}

UpdateCacheDeviceOperation::spec_return_value_t UpdateCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.cache.tensor_spec();
}

Tensor UpdateCacheDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t& args) {
    return args.cache;
}

UpdateCacheDeviceOperation::topology_return_value_t UpdateCacheDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {args.cache.tensor_topology()};
}

ttsl::hash::hash_t UpdateCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& params, const tensor_args_t& args) {
    return operation::hash_operation<UpdateCacheDeviceOperation>(params, args);
}

ProgramDescriptor UpdateCacheProgramFactory::create_descriptor(
    const UpdateCacheParams& params,
    const UpdateCacheInputs& args,
    Tensor&,
    const std::optional<MeshCoordinate>& coord) {
    TT_FATAL(coord.has_value(), "Packed cache update requires a mesh coordinate");
    auto* device = args.input.device();
    const auto& input = args.input.padded_shape();
    const auto& cache = args.cache.padded_shape();
    const uint32_t tiles = input[1] * (params.request_local_tokens / 32) * (input[3] / 32);
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t cores = std::min<uint32_t>(tiles, grid.x * grid.y);
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < cores; ++i) {
        const CoreCoord core{i % grid.x, i / grid.x};
        ranges.emplace_back(core, core);
    }
    const CoreRangeSet core_set(ranges);
    const auto format = datatype_to_dataformat_converter(args.input.dtype());
    const uint32_t page_bytes = args.input.buffer()->page_size();
    KernelDescriptor kernel;
    kernel.kernel_source = "ttnn/cpp/ttnn/operations/experimental/gemma4/device/kernels/update_cache.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = core_set;
    kernel.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    kernel.compile_time_args = {
        params.request_index,
        params.request_local_tokens / 32,
        input[2] / 32,
        input[3] / 32,
        cache[2] / 32,
        cache[1],
        cache[0],
        params.layer_idx,
        params.num_layers,
        device->shape()[params.cluster_axis],
        (*coord)[params.cluster_axis],
        page_bytes};
    TensorAccessorArgs(args.input.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(args.cache.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(args.descriptors.buffer()).append_to(kernel.compile_time_args);
    kernel.emplace_common_runtime_args({args.input.buffer(), args.cache.buffer(), args.descriptors.buffer()});
    for (uint32_t i = 0; i < cores; ++i) {
        kernel.emplace_runtime_args(CoreCoord{i % grid.x, i / grid.x}, {tiles * i / cores, tiles * (i + 1) / cores});
    }
    ProgramDescriptor result;
    result.cbs.push_back(CBDescriptor{
        .total_size = page_bytes,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{.buffer_index = 0, .data_format = format, .page_size = page_bytes}}},
    });
    result.kernels.push_back(std::move(kernel));
    return result;
}

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const Tensor& request_descriptors,
    uint32_t request_index,
    uint32_t request_local_tokens,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis) {
    return ttnn::device_operation::launch<UpdateCacheDeviceOperation>(
        {request_index, request_local_tokens, layer_idx, num_layers, cluster_axis},
        {cache, input, request_descriptors});
}

}  // namespace ttnn::experimental::gemma4
