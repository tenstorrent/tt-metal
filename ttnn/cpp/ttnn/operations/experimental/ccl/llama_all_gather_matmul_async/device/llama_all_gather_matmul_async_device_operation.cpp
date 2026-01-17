// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

LlamaAllGatherMatmulAsyncDeviceOperation::program_factory_t
LlamaAllGatherMatmulAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return LlamaAllGatherMatmulAsyncProgramFactory{};
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

LlamaAllGatherMatmulAsyncDeviceOperation::spec_return_value_t
LlamaAllGatherMatmulAsyncDeviceOperation::compute_output_specs(
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
    ttnn::TensorSpec matmul_output_specs =
        ttnn::prim::MatmulDeviceOperation::compute_output_specs(args.matmul_struct, {{input0, input1}, {}})[0];

    return LlamaAllGatherMatmulAsyncResultSpec{.mm = matmul_output_specs, .aggregated = aggregated_tensor_spec};
}

LlamaAllGatherMatmulAsyncDeviceOperation::tensor_return_value_t
LlamaAllGatherMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Create aggregated tensor internally with exact same specs as pytest
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;

    auto specs = compute_output_specs(args, tensor_args);
    ttnn::Tensor aggregated_tensor = create_device_tensor(specs.aggregated, input0.device());

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = ttnn::prim::MatmulDeviceOperation::create_output_tensors(
        args.matmul_struct, {{aggregated_tensor, input1}, {}})[0];

    return LlamaAllGatherMatmulAsyncResult{.mm = matmul_output_tensor, .aggregated = aggregated_tensor};
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

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation::tensor_return_value_t llama_all_gather_matmul_async(
    const Tensor& input0,
    const Tensor& input1,
    const Tensor& intermediate_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& global_semaphore,
    const std::optional<tt::tt_metal::MemoryConfig>& ag_memory_config,
    const std::optional<tt::tt_metal::MemoryConfig>& mm_memory_config,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const DataType> dtype,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    using OperationType = ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation;
    tt::tt_fabric::Topology usable_topology = ttnn::ccl::get_usable_topology(input0, topology, cluster_axis);

    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather-replicate invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input0.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input0);

    auto matmul_struct = ttnn::prim::create_matmul_attributes(
        input0,
        input1,
        /*parameters=*/
        ttnn::prim::MatmulParams{
            program_config,
            /*bcast_batch=*/std::nullopt,
            mm_memory_config.value_or(input0.memory_config()),
            dtype.value_or(input0.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            /*user_core_coord=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*user_run_batched=*/false,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/global_cb},
        {});

    auto operation_attributes = ttnn::experimental::prim::LlamaAllGatherMatmulAsyncParams(
        matmul_struct,
        devices,
        gather_dim,
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        ag_memory_config.value_or(input0.memory_config()),
        usable_topology,
        global_semaphore,
        sub_device_id,
        cluster_axis);
    auto tensor_args = ttnn::experimental::prim::LlamaAllGatherMatmulAsyncInputs{
        .input0 = input0, .input1 = input1, .intermediate = intermediate_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
