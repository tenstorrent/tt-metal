// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"

#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::ccl {
namespace llama_all_gather_matmul_async {

LlamaAllGatherMatmulAsyncDeviceOperation::program_factory_t
LlamaAllGatherMatmulAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return program::LlamaAllGatherMatmulAsyncProgramFactory{};
}

void LlamaAllGatherMatmulAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LlamaAllGatherMatmulAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input0 = tensor_args.input0;
    const auto& page_size = input0.buffer()->page_size();

    TT_FATAL(page_size % input0.buffer()->alignment() == 0, "All Gather Replicate currently requires aligned pages");
    TT_FATAL(input0.storage_type() == StorageType::DEVICE, "Operands to llama_all_gather_matmul need to be on device!");
    TT_FATAL(
        input0.buffer() != nullptr, "Operands to llama_all_gather_matmul need to be allocated in buffers on device!");
    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    TT_FATAL(
        args.num_links <= input0.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input0.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input0.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input0.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input0.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input0.memory_config().memory_layout());
}

spec_return_value_t LlamaAllGatherMatmulAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;

    auto intermediate_shape = input0.padded_shape();
    intermediate_shape[-1] = intermediate_shape[-1] * args.ring_size;
    auto intermediate_shard_shape = args.output_memory_config.shard_spec()->shape;
    TensorSpec intermediate_tensor_spec = TensorSpec(
        intermediate_shape,
        TensorLayout(input0.dtype(), input0.tensor_spec().page_config(), args.output_memory_config));

    // Calculate aggregated tensor shape and shard specs
    auto aggregated_shape = intermediate_shape;
    aggregated_shape[-1] = intermediate_shape[-1] * 60;

    auto aggregated_shard_shape = intermediate_shard_shape;
    aggregated_shard_shape[1] = intermediate_shard_shape[1] * args.ring_size;

    // Create aggregated tensor memory config
    MemoryConfig aggregated_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            ttnn::CoreRangeSet({ttnn::CoreRange(ttnn::CoreCoord(1, 0), ttnn::CoreCoord(6, 9))}),  // MCAST_CRS
            aggregated_shard_shape,
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    TensorSpec aggregated_tensor_spec = TensorSpec(
        aggregated_shape, TensorLayout(input0.dtype(), input0.tensor_spec().page_config(), aggregated_mem_config));

    // Matmul output spec - using aggregated tensor as input to matmul
    ttnn::TensorSpec matmul_output_specs = args.matmul_struct.compute_output_specs({input0, input1}, {})[0];

    return spec_return_value_t{.mm = matmul_output_specs, .aggregated = aggregated_tensor_spec};
}

tensor_return_value_t LlamaAllGatherMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Create aggregated tensor internally with exact same specs as pytest
    // const auto& intermediate_tensor = input_tensors[2];
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;

    auto specs = compute_output_specs(args, tensor_args);
    ttnn::Tensor aggregated_tensor = create_device_tensor(specs.aggregated, input0.device());

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = args.matmul_struct.create_output_tensors({aggregated_tensor, input1})[0];

    return tensor_return_value_t{.mm = matmul_output_tensor, .aggregated = aggregated_tensor};
}

tt::stl::hash::hash_t LlamaAllGatherMatmulAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;

    auto input0_shape = input0.padded_shape();
    auto input0_memory_layout = input0.layout();
    auto input0_dtype = input0.dtype();
    auto input0_memory_config = input0.memory_config();

    auto input1_shape = input1.padded_shape();
    auto input1_memory_layout = input1.layout();
    auto input1_dtype = input1.dtype();
    auto input1_memory_config = input1.memory_config();

    auto intermediate_shape = input1.padded_shape();
    auto intermediate_memory_layout = input1.layout();
    auto intermediate_dtype = input1.dtype();
    auto intermediate_memory_config = input1.memory_config();

    return tt::tt_metal::operation::hash_operation<LlamaAllGatherMatmulAsyncDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_memory_config,
        args.topology,
        args.cluster_axis,
        input0_shape,
        input0_memory_layout,
        input0_dtype,
        input0_memory_config,
        input1_shape,
        input1_memory_layout,
        input1_dtype,
        input1_memory_config,
        intermediate_shape,
        intermediate_memory_layout,
        intermediate_dtype,
        intermediate_memory_config);
}

std::tuple<
    LlamaAllGatherMatmulAsyncDeviceOperation::operation_attributes_t,
    LlamaAllGatherMatmulAsyncDeviceOperation::tensor_args_t>
LlamaAllGatherMatmulAsyncDeviceOperation::invoke(
    const Tensor& input0,
    const Tensor& input1,
    const Tensor& intermediate_tensor,
    const std::vector<IDevice*>& devices,
    int32_t dim,
    size_t num_links,
    size_t ring_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& global_semaphore,
    const operations::matmul::Matmul& matmul_struct,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tensor_return_value_t>& preallocated_output_tensors) {
    return {
        operation_attributes_t(
            matmul_struct,
            devices,
            dim,
            num_links,
            ring_size,
            output_memory_config,
            topology,
            global_semaphore,
            sub_device_id,
            cluster_axis),
        tensor_args_t{
            .input0 = input0,
            .input1 = input1,
            .intermediate = intermediate_tensor,
            .preallocated_outputs = preallocated_output_tensors}};
}

}  // namespace llama_all_gather_matmul_async
}  // namespace ttnn::operations::experimental::ccl
