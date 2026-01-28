// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_device_operation.hpp"
#include "all_reduce_async_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

AllReduceAsyncDeviceOperation::program_factory_t AllReduceAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return AllReduceAsyncMeshWorkloadFactory{};
}

void AllReduceAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AllReduceAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();

    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM),
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllReduceAsync currently requires aligned pages");
    TT_FATAL(
        args.ring_size % 2 == 0,
        "AllReduceAsync currently only supports even number of blocks in the reduction kernel.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(buffer_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(buffer_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    TT_FATAL(
        args.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor{}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout());
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        args.output_mem_config.memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec()->grid.contains(args.output_mem_config.shard_spec()->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume =
        args.output_mem_config.shard_spec()->shape[0] * args.output_mem_config.shard_spec()->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec()->shape[0] * buffer_tensor.memory_config().shard_spec()->shape[1];
    TT_FATAL(
        output_shard_shape_volume * args.ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * args.ring_size,
        buffer_shard_shape_volume);
}

AllReduceAsyncDeviceOperation::spec_return_value_t AllReduceAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    tt::tt_metal::TensorLayout output_tensor_layout =
        tt::tt_metal::TensorLayout(args.dtype, input_tensor.tensor_spec().page_config(), args.output_mem_config);

    return TensorSpec(shape, output_tensor_layout);
}

AllReduceAsyncDeviceOperation::tensor_return_value_t AllReduceAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t AllReduceAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllReduceAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllReduceAsyncDeviceOperation>(
        args.num_links,
        args.ring_size,
        args.dtype,
        args.output_mem_config,
        args.topology,
        args.use_noc1_only,
        args.use_optimal_ccl_for_llama,
        args.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllReduceAsyncDeviceOperation::tensor_return_value_t all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    uint32_t cluster_axis,
    MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    std::optional<DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    using OperationType = ttnn::experimental::prim::AllReduceAsyncDeviceOperation;
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-reduce invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    auto operation_attributes = OperationType::operation_attributes_t(
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        dtype.value_or(input_tensor.dtype()),
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        use_noc1_only,
        use_optimal_ccl_for_llama,
        cluster_axis,
        &mesh_device);
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .buffer_tensor = buffer_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
