// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
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

ttnn::device_operation::CachedProgram<MeshPartitionDeviceOperation::MeshPartition::shared_variables_t>
MeshPartitionDeviceOperation::MeshPartition::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
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

    // Use the new prim::slice operation
    using SliceOp = ttnn::operations::data_movement::SliceDeviceOperation;

    auto [slice_attrs, slice_tensor_args] = SliceOp::invoke(
        tensor_args.input_tensor,
        begins,
        ends,
        strides,
        operation_attributes.output_mem_config,
        false,         // use_tensor_args
        std::nullopt,  // start_tensor
        std::nullopt,  // end_tensor
        std::nullopt,  // slice_dim
        std::nullopt,  // num_devices
        std::nullopt,  // sub_core_grids
        std::nullopt   // preallocated_output
    );

    SliceOp::validate_on_program_cache_miss(slice_attrs, slice_tensor_args);

    auto program_factory = SliceOp::select_program_factory(slice_attrs, slice_tensor_args);

    // Create the cached program by visiting the variant
    // We need to use a common return type, so we extract just the program
    Program program = std::visit(
        [&](auto&& factory) -> Program {
            auto cached_prog = factory.create(slice_attrs, slice_tensor_args, tensor_return_value);
            return std::move(cached_prog.program);
        },
        program_factory);

    // -- building the return value -----------------------------------
    // Note: Runtime argument override is not implemented for slice in mesh_partition yet
    // The override callback is left empty for now
    shared_variables_t vars{
        .override_runtime_arguments_callback = OverrideRuntimeArgsCallback{},
        .slice_attrs = slice_attrs,
        .program_factory = program_factory};

    return {std::move(program), std::move(vars)};
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
        // Runtime argument override not yet supported for TMP slice operation in mesh_partition
        if (shared_variables.override_runtime_arguments_callback) {
            shared_variables.override_runtime_arguments_callback(
                nullptr, program, input_tensors, input_tensor_options, output_tensors);
        }
    }
}

}  // namespace ttnn::operations::ccl
