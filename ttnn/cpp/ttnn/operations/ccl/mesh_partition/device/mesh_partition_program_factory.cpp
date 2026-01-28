// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tuple>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
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

namespace {

using SliceOp = ttnn::prim::SliceDeviceOperation;

// Helper function to compute slice parameters for a given mesh coordinate
auto compute_slice_parameters(
    const MeshPartitionDeviceOperation::operation_attributes_t& operation_attributes,
    const MeshPartitionDeviceOperation::tensor_args_t& tensor_args,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    const auto& input_tensor = tensor_args.input_tensor;

    const uint32_t cluster_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);
    uint32_t cluster_index =
        detail::get_cluster_axis_index(input_tensor.device()->get_view(), mesh_coordinate, operation_attributes);

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

    auto slice_arg_func =
        [](auto input, auto slice_start, auto slice_end, auto step, auto output_mem_config, auto use_tensor_args) {
            return std::make_tuple(
                SliceOp::operation_attributes_t{
                    .slice_start = std::move(slice_start),
                    .slice_end = std::move(slice_end),
                    .step = std::move(step),
                    .output_mem_config = std::move(output_mem_config),
                    .use_tensor_args = use_tensor_args,
                    .slice_dim = std::nullopt,
                    .num_devices = std::nullopt,
                    .sub_core_grids = std::nullopt},
                SliceOp::tensor_args_t{
                    .input = std::move(input),
                    .start_tensor = std::nullopt,
                    .end_tensor = std::nullopt,
                    .preallocated_output = std::nullopt});
        };
    return slice_arg_func(
        tensor_args.input_tensor,
        begins,
        ends,
        strides,
        operation_attributes.output_mem_config,
        false  // use_tensor_args
    );
}

}  // anonymous namespace

ttnn::device_operation::CachedProgram<MeshPartitionDeviceOperation::MeshPartition::shared_variables_t>
MeshPartitionDeviceOperation::MeshPartition::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto [slice_attrs, slice_tensor_args] =
        compute_slice_parameters(operation_attributes, tensor_args, mesh_coordinate);

    SliceOp::validate_on_program_cache_miss(slice_attrs, slice_tensor_args);
    auto program_factory = SliceOp::select_program_factory(slice_attrs, slice_tensor_args);
    auto program_and_shared_variables = std::visit(
        [&](auto&& factory) -> std::pair<Program, SliceSharedVariables> {
            auto cached_program = factory.create(slice_attrs, slice_tensor_args, tensor_return_value);
            return {std::move(cached_program.program), std::move(cached_program.shared_variables)};
        },
        program_factory);

    shared_variables_t vars{
        .slice_program_factory = program_factory,
        .slice_shared_variables = std::move(program_and_shared_variables.second)};
    return {std::move(program_and_shared_variables.first), std::move(vars)};
}

void MeshPartitionDeviceOperation::MeshPartition::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(range);

        // Get the mesh coordinate from the range (assuming single device per range)
        auto mesh_coordinate = *range.begin();
        auto [slice_attrs, slice_tensor_args] =
            compute_slice_parameters(operation_attributes, tensor_args, mesh_coordinate);

        // Visit the program factory variant and use std::get to extract the matching shared_variables
        std::visit(
            [&](auto&& program_factory) {
                using Factory = std::decay_t<decltype(program_factory)>;
                using SharedVars = typename Factory::shared_variables_t;

                auto& slice_shared_vars = std::get<SharedVars>(shared_variables.slice_shared_variables);
                auto cached_proxy_program = Factory::cached_program_t::proxy(program, slice_shared_vars);
                program_factory.override_runtime_arguments(
                    cached_proxy_program, slice_attrs, slice_tensor_args, tensor_return_value);
            },
            shared_variables.slice_program_factory);
    }
}

}  // namespace ttnn::operations::ccl
