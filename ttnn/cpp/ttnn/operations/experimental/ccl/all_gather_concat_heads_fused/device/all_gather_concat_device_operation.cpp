// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat_device_operation.hpp"

#include <algorithm>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::experimental::ccl {

AllGatherConcatDeviceOperation::program_factory_t AllGatherConcatDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllGatherConcatProgram{};
}

void AllGatherConcatDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void AllGatherConcatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(operation_attributes.semaphore.has_value(), "Semaphore is required for AllGatherConcat operation");

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
    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

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
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.padded_shape();
    auto num_heads = operation_attributes.num_heads;
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
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::Layout::TILE, operation_attributes.output_mem_config));
}

AllGatherConcatDeviceOperation::tensor_return_value_t AllGatherConcatDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t AllGatherConcatDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.padded_shape();
    auto input_memory_layout = input_tensor.layout();
    auto input_dtype = input_tensor.dtype();
    auto input_memory_config = input_tensor.memory_config();

    return tt::tt_metal::operation::hash_operation<AllGatherConcatDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        operation_attributes.num_heads,
        operation_attributes.use_noc1_only);
}

std::tuple<AllGatherConcatDeviceOperation::operation_attributes_t, AllGatherConcatDeviceOperation::tensor_args_t>
AllGatherConcatDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    uint32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    uint32_t num_heads,
    bool use_noc1_only,
    const MemoryConfig& memory_config,
    std::optional<uint32_t> num_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    uint32_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return {
        operation_attributes_t{
            .dim = static_cast<uint32_t>(gather_dim),
            .num_links = num_links.value_or(1),
            .ring_size = ring_size,
            .output_mem_config = memory_config,
            .topology = topology,
            .semaphore = semaphore,
            .sub_device_id = sub_device_id,
            .num_heads = num_heads,
            .use_noc1_only = use_noc1_only,
            .cluster_axis = cluster_axis},
        tensor_args_t{.input_tensor = input_tensor, .buffer_tensor = buffer_tensor}};
}

}  // namespace ttnn::operations::experimental::ccl
