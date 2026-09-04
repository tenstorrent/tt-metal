// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tuple>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace ttnn::operations::ccl {
namespace detail {

// True for a slice program factory that has migrated to the Metal 2.0 spec concept. Keyed on the
// entry point rather than on a list of factory names, so each further slice factory that migrates
// needs no change here.
template <typename T>
concept IsSliceSpecFactory = requires { &T::create_program_artifacts; };

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

    auto input_shape = input_tensor.padded_shape();
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
    Program program = std::visit(
        [&](auto&& factory) -> Program {
            using Factory = std::decay_t<decltype(factory)>;
            // Slice's factories are migrating to Metal 2.0 one at a time, so this visit spans both
            // shapes. A Metal 2.0 factory has no create_descriptor to call -- declaring one would
            // re-classify it as a descriptor factory and its spec entry point would never run -- so
            // it is built from its spec instead. The remaining factories take the descriptor path
            // unchanged, and this branch retires when the last of them converts.
            if constexpr (detail::IsSliceSpecFactory<Factory>) {
                auto artifacts = Factory::create_program_artifacts(slice_attrs, slice_tensor_args, tensor_return_value);
                Program spec_program =
                    tt::tt_metal::experimental::MakeProgramFromSpec(*tensor_args.input_tensor.device(), artifacts.spec);
                tt::tt_metal::experimental::SetProgramRunArgs(spec_program, artifacts.run_params);
                return spec_program;
            } else {
                auto descriptor = Factory::create_descriptor(slice_attrs, slice_tensor_args, tensor_return_value);
                return Program{descriptor};
            }
        },
        program_factory);

    return {std::move(program), shared_variables_t{.slice_program_factory = program_factory}};
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

        // Re-apply this coord's per-dispatch state to the cached Program, through the same path the
        // slice op uses for whichever factory built it. Buffer sizing is not re-applied on a hit, so
        // any sizing that varies across calls must be in compute_program_hash().
        std::visit(
            [&](auto&& factory) {
                using Factory = std::decay_t<decltype(factory)>;
                if constexpr (detail::IsSliceSpecFactory<Factory>) {
                    tt::tt_metal::experimental::UpdateProgramRunArgs(
                        program,
                        Factory::override_runtime_arguments(
                            slice_attrs, slice_tensor_args, tensor_return_value, mesh_coordinate));
                } else {
                    ttnn::prim::patch_slice_program_addresses(
                        program,
                        shared_variables.slice_program_factory,
                        slice_attrs,
                        slice_tensor_args,
                        tensor_return_value);
                }
            },
            shared_variables.slice_program_factory);
    }
}

}  // namespace ttnn::operations::ccl
