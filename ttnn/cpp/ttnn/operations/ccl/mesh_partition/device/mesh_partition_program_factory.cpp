// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::ccl {
namespace detail {
uint32_t get_cluster_axis_index(
    const ttnn::MeshDeviceView& mesh_view,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MeshPartitionDeviceOperation::operation_attributes_t& operation_attributes) {
    return operation_attributes.cluster_axis.has_value()
               ? ((operation_attributes.cluster_axis.value() == 0) ? mesh_coordinate[0] : mesh_coordinate[1])
               : common::get_linearized_index(mesh_coordinate, mesh_view);
}
}  // namespace detail

MeshPartitionDeviceOperation::MeshPartition::cached_mesh_workload_t
MeshPartitionDeviceOperation::MeshPartition::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MeshPartitionDeviceOperation::MeshPartition::shared_variables_t>
MeshPartitionDeviceOperation::MeshPartition::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;

    const uint32_t cluster_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);
    uint32_t cluster_index =
        detail::get_cluster_axis_index(input_tensor.mesh_device()->get_view(), mesh_coordinate, operation_attributes);
    TT_FATAL(
        cluster_index < cluster_size,
        "cluster_index ({}) must be less than cluster_size ({})",
        cluster_index,
        cluster_size);

    auto input_shape = input_tensor.logical_shape();
    uint32_t dim = operation_attributes.dim;
    uint32_t rank = input_shape.size();
    auto partitioned_dim_size = input_shape[dim] / cluster_size;
    uint64_t begin_pos = static_cast<uint64_t>(cluster_index) * partitioned_dim_size;
    TT_FATAL(
        begin_pos <= std::numeric_limits<uint32_t>::max() - partitioned_dim_size,
        "Integer overflow: cluster_index ({}) * partitioned_dim_size ({}) = {} exceeds uint32_t max",
        cluster_index,
        partitioned_dim_size,
        begin_pos);

    auto begins = ttnn::Shape(std::vector<uint32_t>(rank, 0));
    auto ends = input_shape;
    auto strides = ttnn::Shape(std::vector<uint32_t>(rank, 1));

    begins[dim] = static_cast<uint32_t>(begin_pos);
    ends[dim] = begins[dim] + partitioned_dim_size;

    TT_FATAL(
        ends[dim] <= input_shape[dim],
        "Slice bounds error: ends[{}] ({}) exceeds input_shape[{}] ({})",
        dim,
        ends[dim],
        dim,
        input_shape[dim]);

    log_debug(
        tt::LogOp,
        "Slice at ({}, {}) will have begins {}, ends {}, step {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        begins,
        ends,
        strides);

    auto slice_op = ttnn::operations::data_movement::SliceDeviceOperation{
        .slice_start = begins,
        .slice_end = ends,
        .step = strides,
        .output_mem_config = operation_attributes.output_mem_config};

    auto input_tensors = std::vector<ttnn::Tensor>{tensor_args.input_tensor};
    auto output_tensors = std::vector<ttnn::Tensor>{tensor_return_value};
    auto optional_output_tensors = std::vector<std::optional<ttnn::Tensor>>{std::nullopt};
    slice_op.validate_with_output_tensors(input_tensors, optional_output_tensors);

    auto cached_program = slice_op.create_program(input_tensors, output_tensors);
    TT_FATAL(
        cached_program.override_runtime_arguments_callback.has_value(),
        "override_runtime_arguments_callback is not set for program at mesh coordinate ({}, {})",
        mesh_coordinate[0],
        mesh_coordinate[1]);

    // -- building the return value -----------------------------------
    shared_variables_t vars{
        // if the optional holds a callback, move it; otherwise construct an empty
        // std::function (== "no-op")
        .override_runtime_arguments_callback = cached_program.override_runtime_arguments_callback.value_or(
            OverrideRuntimeArgsCallback{}),  //   ^ empty functor
        .slice_op = slice_op};

    return {std::move(cached_program.program), std::move(vars)};
}

void MeshPartitionDeviceOperation::MeshPartition::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    std::vector<tt::tt_metal::Tensor> input_tensors{tensor_args.input_tensor};
    std::vector<std::optional<const tt::tt_metal::Tensor>> input_tensor_options{};
    std::vector<tt::tt_metal::Tensor> output_tensors{tensor_return_value};
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        shared_variables.override_runtime_arguments_callback(
            (const void*)&shared_variables.slice_op, program, input_tensors, input_tensor_options, output_tensors);
    }
}

}  // namespace ttnn::operations::ccl
