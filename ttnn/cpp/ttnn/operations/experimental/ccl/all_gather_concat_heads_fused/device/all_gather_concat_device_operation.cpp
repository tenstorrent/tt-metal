// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::all_gather_concat_heads_fused {

AllGatherConcatDeviceOperation::program_factory_t AllGatherConcatDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::AllGatherConcatMeshWorkloadFactory{};
}

void AllGatherConcatDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AllGatherConcatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();
    const auto input_core_ranges = input_tensor.buffer()->shard_spec().grid().ranges();
    const auto& padded_input_shape = input_tensor.padded_shape();
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "All Gather currently requires aligned pages");
    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM),
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but is {}", args.num_links);
    TT_FATAL(
        args.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by {} links are parallelized over {} rows",
        args.num_links,
        input_tensor.device()->compute_with_storage_grid_size().y);

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        input_core_ranges[0].start_coord.x == 1 && input_core_ranges[0].end_coord.x == 3 &&
            input_core_ranges[0].start_coord.y == 0 && input_core_ranges[0].end_coord.y == 1 &&
            input_core_ranges[1].start_coord.x == 1 && input_core_ranges[1].end_coord.x == 2 &&
            input_core_ranges[1].start_coord.y == 2 && input_core_ranges[1].end_coord.y == 2,
        "Unsupported input core ranges!");
    CoreCoord grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(grid_size.x >= 3 && grid_size.y >= 3, "Input core grid out of bound!");
    TT_FATAL(
        padded_input_shape[0] == 1 && padded_input_shape[1] == 8 && padded_input_shape[3] == 128,
        "Unsupported input shape, should be [1, 8, 32, 128] or [1, 8, 8, 128]!");
}

AllGatherConcatDeviceOperation::spec_return_value_t AllGatherConcatDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.padded_shape();  // TODO: Replace with logical_shape()
    auto num_heads = args.num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    uint32_t batch_size = 32;
    batch = std::max(batch, batch_size);
    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::Layout::TILE, args.output_mem_config));
}

AllGatherConcatDeviceOperation::tensor_return_value_t AllGatherConcatDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t AllGatherConcatDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherConcatDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllGatherConcatDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_mem_config,
        args.topology,
        args.num_heads,
        args.use_noc1_only,
        args.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::all_gather_concat_heads_fused

namespace ttnn::prim {

ttnn::operations::experimental::ccl::all_gather_concat_heads_fused::AllGatherConcatDeviceOperation::
    tensor_return_value_t
    all_gather_concat(
        const Tensor& input_tensor,
        Tensor& buffer_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& global_semaphore,
        uint32_t num_heads,
        const MemoryConfig& memory_config,
        bool use_noc1_only,
        std::optional<uint32_t> num_links,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    using OperationType =
        ttnn::operations::experimental::ccl::all_gather_concat_heads_fused::AllGatherConcatDeviceOperation;
    const auto& mesh_view = mesh_device.get_view();
    uint32_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    auto operation_attributes = OperationType::operation_attributes_t(
        static_cast<uint32_t>(gather_dim),
        num_links.value_or(1),
        num_devices,
        memory_config,
        topology,
        global_semaphore,
        sub_device_id,
        num_heads,
        use_noc1_only,
        cluster_axis);

    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .buffer_tensor = buffer_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
