// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tuple>
#include <vector>
#include <tracy/Tracy.hpp>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <tt-metalium/host_api.hpp>

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
    ZoneNamedN(__tracy_scoped_zone, "HostProfile::partition_compute_slice_params", ([] {
                   static const bool enabled = std::getenv("TT_METAL_HOST_PROFILE_ZONES") != nullptr;
                   return enabled;
               }()));
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
            auto descriptor = Factory::create_descriptor(slice_attrs, slice_tensor_args, tensor_return_value);
            // Only MeshPartition opts in to this ABI. Standalone Slice retains its unique
            // address slot and dynamic scalar patching. Keep slot 0 reserved so every work
            // split field keeps the exact offset emitted by the underlying Slice factory.
            const auto use_common_address = [&](uint32_t kernel_idx, tt::tt_metal::Buffer* buffer) {
                TT_FATAL(kernel_idx < descriptor.kernels.size(), "MeshPartition address kernel is missing");
                auto& kernel = descriptor.kernels[kernel_idx];
                TT_FATAL(kernel.common_runtime_args.empty(), "MeshPartition address kernel already has common args");
                TT_FATAL(
                    kernel.common_buffer_bindings.empty(), "MeshPartition address kernel already has common bindings");
                for (const auto& binding : kernel.buffer_bindings) {
                    TT_FATAL(
                        binding.arg_idx == 0 && binding.buffer == buffer,
                        "MeshPartition expected only its operand binding at unique slot 0");
                }
                kernel.buffer_bindings.clear();
                for (auto& [core, args] : kernel.runtime_args) {
                    args.at(0) = 0;
                }
                kernel.defines.emplace_back("MESH_PARTITION_COMMON_ADDRESS", "1");
                kernel.emplace_common_runtime_args({buffer});
            };
            if constexpr (std::is_same_v<Factory, ttnn::prim::SliceTileProgramFactory>) {
                use_common_address(1, tensor_return_value.buffer());
            } else if constexpr (std::is_same_v<Factory, ttnn::prim::SliceRmProgramFactory>) {
                use_common_address(0, tensor_args.input_tensor.buffer());
                use_common_address(1, tensor_return_value.buffer());
            }
            return Program{descriptor};
        },
        program_factory);

    // Each coordinate owns its work split and program. Both addresses are common in the
    // tiled/RM variants; owning accessors fetch live storage after enqueue/trace retargeting.
    const bool common_addresses = std::holds_alternative<ttnn::prim::SliceTileProgramFactory>(program_factory) ||
                                  std::holds_alternative<ttnn::prim::SliceRmProgramFactory>(program_factory);
    std::optional<shared_variables_t::AddressPlan> address_plan;
    if (common_addresses) {
        address_plan.emplace(shared_variables_t::AddressPlan{
            .reader = tt::tt_metal::KernelRuntimeArgsAccessor(program, 0),
            .writer = tt::tt_metal::KernelRuntimeArgsAccessor(program, 1)});
    }
    return {
        std::move(program),
        shared_variables_t{
            .slice_program_factory = program_factory,
            .slice_attributes = std::move(slice_attrs),
            .address_plan = std::move(address_plan)}};
}

void MeshPartitionDeviceOperation::MeshPartition::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    static const bool profile_phases = std::getenv("TT_METAL_HOST_PROFILE_PHASES") != nullptr;
    // Distributed buffers have a common virtual address across coordinates. Resolve these
    // once per invocation, while each owning accessor retrieves its coordinate's live storage.
    const uint32_t input_address = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_address = tensor_return_value.buffer()->address();
    for (auto& [range, shared_variables] : cached_workload.shared_variables) {
        if (shared_variables.address_plan) {
            const auto& plan = *shared_variables.address_plan;
            ZoneNamedN(mesh_partition_addresses, "HostProfile::mesh_partition_address_plan", profile_phases);
            plan.reader.common_runtime_args().at(0) = input_address;
            plan.writer.common_runtime_args().at(0) = output_address;
        } else {
            // Keep sharded CB and any future strided factory on the existing helper.
            const SliceOp::tensor_args_t slice_tensor_args{
                .input = tensor_args.input_tensor,
                .start_tensor = std::nullopt,
                .end_tensor = std::nullopt,
                .preallocated_output = std::nullopt};
            ttnn::prim::patch_slice_program_addresses(
                cached_workload.workload.get_programs().at(range),
                shared_variables.slice_program_factory,
                shared_variables.slice_attributes,
                slice_tensor_args,
                tensor_return_value);
        }
    }
}

}  // namespace ttnn::operations::ccl
