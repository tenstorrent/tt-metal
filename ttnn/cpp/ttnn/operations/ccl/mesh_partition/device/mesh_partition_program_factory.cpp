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
            return Program{descriptor};
        },
        program_factory);

    // Each coordinate owns a distinct ProgramImpl. The workload key includes tensor specs,
    // partition attributes and coordinates, so the descriptor's scalar work split is invariant.
    // All interleaved factories bind addresses at slot 0; tiled readers use common slot 0.
    // Compress consecutive active cores in each matrix row, without retaining payload pointers.
    const auto collect_address_runs = [](const tt::tt_metal::KernelRuntimeArgsAccessor& accessor) {
        std::vector<shared_variables_t::AddressRun> runs;
        const auto& args = accessor.runtime_args();
        for (uint32_t x = 0; x < args.size(); ++x) {
            for (uint32_t y = 0; y < args[x].size();) {
                if (args[x][y].size() == 0 || args[x][y][0] == 0) {
                    ++y;
                    continue;
                }
                const uint32_t begin = y++;
                while (y < args[x].size() && args[x][y].size() != 0 && args[x][y][0] != 0) {
                    ++y;
                }
                runs.push_back({x, begin, y});
            }
        }
        return runs;
    };
    const bool tiled = std::holds_alternative<ttnn::prim::SliceTileProgramFactory>(program_factory);
    const bool rm = std::holds_alternative<ttnn::prim::SliceRmProgramFactory>(program_factory) ||
                    std::holds_alternative<ttnn::prim::SliceRmStrideProgramFactory>(program_factory);
    std::optional<shared_variables_t::AddressPlan> address_plan;
    if (tiled || rm) {
        address_plan.emplace(shared_variables_t::AddressPlan{
            .reader = tt::tt_metal::KernelRuntimeArgsAccessor(program, 0),
            .writer = tt::tt_metal::KernelRuntimeArgsAccessor(program, 1),
            .reader_runs = {},
            .writer_runs = {},
            .common_reader_address = tiled});
        address_plan->writer_runs = collect_address_runs(address_plan->writer);
        if (rm) {
            address_plan->reader_runs = collect_address_runs(address_plan->reader);
        }
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
            const auto patch_addresses = [&](const auto& accessor, const auto& runs, uint32_t address) {
                auto& args = accessor.runtime_args();
                for (const auto& run : runs) {
                    auto& row = args.at(run.x);
                    for (uint32_t y = run.y_begin; y < run.y_end; ++y) {
                        row.at(y).at(0) = address;
                    }
                }
            };
            ZoneNamedN(mesh_partition_addresses, "HostProfile::mesh_partition_address_plan", profile_phases);
            patch_addresses(plan.writer, plan.writer_runs, output_address);
            if (plan.common_reader_address) {
                plan.reader.common_runtime_args().at(0) = input_address;
            } else {
                patch_addresses(plan.reader, plan.reader_runs, input_address);
            }
        } else {
            // CB-bound sharded RM keeps its existing address-only helper.
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
