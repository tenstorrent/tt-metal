// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_program_factory.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kNumDirections = 2;

// The names the per-coordinate programs and the cache-hit override share.
struct SliceReshardSpecNames {
    tt::tt_metal::DFBSpecName sender_dfb{"sender"};
    tt::tt_metal::TensorParamName input_param{"input"};
    tt::tt_metal::TensorParamName output_param{"output"};
    tt::tt_metal::experimental::GlobalSemaphoreParamName final_semaphore_param{"final_semaphore"};
    tt::tt_metal::experimental::GlobalSemaphoreParamName barrier_semaphore_param{"barrier_semaphore"};
    tt::tt_metal::experimental::FabricConnectionSpecName connection_spec{"connection"};
};

// The run args every dispatch supplies: the io tensors and the caller-supplied GlobalSemaphores.
tt::tt_metal::ProgramRunArgs make_dispatch_run_args(
    const SliceReshardAsyncParams& args, const Tensor& input_tensor, const Tensor& output_tensor) {
    const SliceReshardSpecNames names;
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.tensor_args.emplace(names.input_param, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(names.output_param, output_tensor.mesh_tensor());
    run_args.advanced_options.global_semaphore_args.emplace(
        names.final_semaphore_param, std::cref(args.final_semaphore));
    run_args.advanced_options.global_semaphore_args.emplace(
        names.barrier_semaphore_param, std::cref(args.barrier_semaphore));
    return run_args;
}

ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram build_slice_reshard_program(
    const SliceReshardAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    using namespace tt::tt_metal;
    const SliceReshardSpecNames names;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    // User specified the cluster-axis. Derive the ring based on the current coordinate
    // and the cluster-axis.
    const auto fabric_node_ids = (args.cluster_axis == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                                          : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);
    const uint32_t ring_size = fabric_node_ids.size();
    const uint32_t ring_index = mesh_coord[args.cluster_axis];

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    if (ring_index != 0) {
        backward_fabric_node_id = fabric_node_ids.at(ring_index - 1);
    }
    if (ring_index != ring_size - 1) {
        forward_fabric_node_id = fabric_node_ids.at(ring_index + 1);
    }

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    TT_FATAL(input_tensor.buffer() != nullptr, "slice_reshard_async: input tensor buffer is null");

    // Get OP Config, topology config
    const uint32_t page_size = static_cast<uint32_t>(input_tensor.buffer()->page_size());
    uint32_t num_sticks_per_outer_dim = input_tensor_shape[1] * input_tensor_shape[2];
    uint32_t input_outer_dim_size = input_tensor_shape[0];
    uint32_t output_outer_dim_size = output_tensor_shape[0];
    bool is_first_device = !backward_fabric_node_id.has_value();
    bool is_last_device = !forward_fabric_node_id.has_value();
    // output coords for this device, in the input space
    uint32_t global_output_outer_dim_start = args.output_dim_offset + (output_outer_dim_size * ring_index);
    uint32_t global_output_outer_dim_end = args.output_dim_offset + (output_outer_dim_size * (ring_index + 1)) - 1;
    // input coords for this device, in the input space
    uint32_t global_input_outer_dim_start = input_outer_dim_size * ring_index;
    uint32_t global_input_outer_dim_end = (input_outer_dim_size * (ring_index + 1)) - 1;

    int32_t backward_device_end = (int32_t)global_input_outer_dim_start - 1;
    uint32_t outer_dims_from_backward = std::max(backward_device_end - (int32_t)global_output_outer_dim_start + 1, 0);
    int32_t forward_device_start = global_input_outer_dim_end + 1;
    uint32_t outer_dims_from_forward = std::max((int32_t)global_output_outer_dim_end - forward_device_start + 1, 0);
    uint32_t outer_dims_to_keep_start =
        std::max((int32_t)global_output_outer_dim_start - (int32_t)global_input_outer_dim_start, 0);
    uint32_t outer_dims_to_keep_end = std::min(
        outer_dims_to_keep_start - outer_dims_from_backward + output_outer_dim_size - 1, input_outer_dim_size - 1);
    int32_t backward_device_output_end =
        std::max((int32_t)global_output_outer_dim_start - 1, (int32_t)args.output_dim_offset - 1);
    uint32_t outer_dims_to_backward =
        is_first_device ? 0 : std::max(backward_device_output_end - (int32_t)global_input_outer_dim_start + 1, 0);
    int32_t forward_device_output_start =
        std::min(global_output_outer_dim_end + 1, output_outer_dim_size * ring_size - 1);
    uint32_t outer_dims_to_forward =
        is_last_device ? 0 : std::max((int32_t)global_input_outer_dim_end - forward_device_output_start + 1, 0);

    // Get worker cores
    CoreCoord core_grid(args.num_links * kNumDirections, 1);
    auto
        [num_cores,
         worker_core_ranges,
         core_group_1,
         core_group_2,
         num_sticks_per_core_group_1,
         num_sticks_per_core_group_2] = tt::tt_metal::split_work_to_cores(core_grid, num_sticks_per_outer_dim * 2);

    // L1 Scratch DFB Creation
    uint32_t l1_scratch_dfb_page_size_bytes = page_size;

    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t dfb_num_pages = 3 * num_sticks_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    ProgramSpec spec{.name = "slice_reshard_async"};

    // DFB for transferring data between reader and writer
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = names.sender_dfb,
        .entry_size = l1_scratch_dfb_page_size_bytes,
        .num_entries = dfb_num_pages,
        .data_format_metadata = df,
    });

    spec.tensor_parameters = {
        TensorParameter{.unique_id = names.input_param, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = names.output_param, .spec = output_tensor.tensor_spec()},
    };
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = names.final_semaphore_param, .memory_type = BufferType::L1},
        {.unique_id = names.barrier_semaphore_param, .memory_type = BufferType::L1},
    };

    tt::tt_metal::experimental::FabricConnectionSpec fabric_connection{
        .unique_id = names.connection_spec,
        .src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord),
    };

    ProgramRunArgs run_args = make_dispatch_run_args(args, input_tensor, output_tensor);

    // KERNEL CREATION
    // Every worker core `{core_idx, 0}` (core_idx = link * kNumDirections + direction) gets its own reader kernel
    // and its own writer kernel.
    uint32_t stick_start_id = 0;
    for (uint32_t link = 0; link < args.num_links; link++) {
        uint32_t num_sticks_to_read = 0;
        for (uint32_t direction = 0; direction < kNumDirections; direction++) {
            const uint32_t core_idx = (link * kNumDirections) + direction;
            CoreCoord core = {core_idx, 0};
            CoreCoord opposite_core = {(link * kNumDirections) + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (core_group_1.contains(core)) {
                num_sticks_to_read = num_sticks_per_core_group_1;
            } else {
                num_sticks_to_read = num_sticks_per_core_group_2;
            }

            const KernelSpecName reader_kernel{"reader_" + std::to_string(core_idx)};
            const KernelSpecName writer_kernel{"writer_" + std::to_string(core_idx)};

            // Reader
            KernelSpec reader_spec{
                .unique_id = reader_kernel,
                .source = std::filesystem::path(
                    "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                    "minimal_default_reader.cpp"),
                .dfb_bindings =
                    {
                        DFBBinding{
                            .dfb_spec_name = names.sender_dfb,
                            .accessor_name = "output",
                            .endpoint_type = DFBEndpointType::PRODUCER,
                        },
                    },
                .tensor_bindings =
                    {
                        TensorBinding{
                            .tensor_parameter_name = names.input_param,
                            .accessor_name = "src",
                        },
                    },
                .compile_time_args =
                    {
                        {"is_first_chip", direction ? is_first_device : is_last_device},
                        {"is_last_chip", direction ? is_last_device : is_first_device},
                        {"direction", direction},
                        {"stick_size", page_size},
                    },
                .runtime_arg_schema =
                    {
                        .runtime_arg_names =
                            {
                                "stick_start_id",
                                "num_sticks_to_read",
                                "input_outer_dim_size",
                                "outer_dims_to_forward",
                                "outer_dims_from_forward",
                                "outer_dims_to_keep_start",
                                "outer_dims_to_keep_end",
                                "num_sticks_per_outer_dim",
                            },
                    },
                .hw_config = ttnn::create_reader_datamovement_config(),
            };
            reader_spec.advanced_options.global_semaphore_bindings = {
                {
                    .global_semaphore_parameter_name = names.final_semaphore_param,
                    .accessor_name = "out_ready_sem",
                },
            };

            KernelRunArgs reader_run_args{
                .kernel = reader_kernel,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    core,
                    {
                        {"stick_start_id", stick_start_id},
                        {"num_sticks_to_read", num_sticks_to_read},
                        {"input_outer_dim_size", input_outer_dim_size},
                        {"outer_dims_to_forward", direction ? outer_dims_to_forward : outer_dims_to_backward},
                        {"outer_dims_from_forward", outer_dims_from_forward},
                        {"outer_dims_to_keep_start", outer_dims_to_keep_start},
                        {"outer_dims_to_keep_end", outer_dims_to_keep_end},
                        {"num_sticks_per_outer_dim", num_sticks_per_outer_dim},
                    }),
            };

            // Writer
            KernelSpec writer_spec{
                .unique_id = writer_kernel,
                .source = std::filesystem::path(
                    "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                    "minimal_default_writer.cpp"),
                .dfb_bindings =
                    {
                        DFBBinding{
                            .dfb_spec_name = names.sender_dfb,
                            .accessor_name = "output",
                            .endpoint_type = DFBEndpointType::CONSUMER,
                        },
                    },
                .tensor_bindings =
                    {
                        TensorBinding{
                            .tensor_parameter_name = names.output_param,
                            .accessor_name = "dst",
                        },
                    },
                .compile_time_args =
                    {
                        {"is_first_chip", direction ? is_first_device : is_last_device},
                        {"is_last_chip", direction ? is_last_device : is_first_device},
                        {"direction", direction},
                    },
                .runtime_arg_schema =
                    {
                        .runtime_arg_names =
                            {
                                "stick_size",
                                "stick_start_id",
                                "num_sticks_to_read",
                                "output_outer_dim_size",
                                "outer_dims_to_forward",
                                "outer_dims_to_keep_start",
                                "outer_dims_to_keep_end",
                                "outer_dims_to_receive",
                                "outer_dims_from_forward",
                                "num_sticks_per_outer_dim",
                                "out_ready_sem_noc0_x",
                                "out_ready_sem_noc0_y",
                                "use_barrier_sem",
                                "barrier_sem_noc0_x",
                                "barrier_sem_noc0_y",
                            },
                    },
                .hw_config = ttnn::create_writer_datamovement_config(),
            };
            writer_spec.advanced_options.global_semaphore_bindings = {
                {
                    .global_semaphore_parameter_name = names.final_semaphore_param,
                    .accessor_name = "out_ready_sem",
                },
                {
                    .global_semaphore_parameter_name = names.barrier_semaphore_param,
                    .accessor_name = "barrier_sem",
                },
            };
            writer_spec.advanced_options.fabric_connection_bindings = {
                {
                    .fabric_connection_spec_name = names.connection_spec,
                    .accessor_name = "connection",
                },
            };

            KernelRunArgs writer_run_args{
                .kernel = writer_kernel,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    core,
                    {
                        {"stick_size", page_size},
                        {"stick_start_id", stick_start_id},
                        {"num_sticks_to_read", num_sticks_to_read},
                        {"output_outer_dim_size", output_outer_dim_size},
                        {"outer_dims_to_forward", direction ? outer_dims_to_forward : outer_dims_to_backward},
                        {"outer_dims_to_keep_start", outer_dims_to_keep_start},
                        {"outer_dims_to_keep_end", outer_dims_to_keep_end},
                        {"outer_dims_to_receive", direction ? outer_dims_from_backward : outer_dims_from_forward},
                        {"outer_dims_from_forward", outer_dims_from_forward},
                        {"num_sticks_per_outer_dim", num_sticks_per_outer_dim},
                        {"out_ready_sem_noc0_x", static_cast<uint32_t>(virtual_core.x)},
                        {"out_ready_sem_noc0_y", static_cast<uint32_t>(virtual_core.y)},
                        {"use_barrier_sem", 1u},
                        {"barrier_sem_noc0_x", static_cast<uint32_t>(virtual_opposite_core.x)},
                        {"barrier_sem_noc0_y", static_cast<uint32_t>(virtual_opposite_core.y)},
                    }),
            };

            // The direction-1 writer sends to the forward neighbor, the direction-0 writer to the backward one.
            const auto& neighbor_fabric_node_id = direction ? forward_fabric_node_id : backward_fabric_node_id;
            if (neighbor_fabric_node_id.has_value()) {
                fabric_connection.targets[core] = {
                    tt::tt_metal::experimental::FabricConnectionTarget{
                        .dst_fabric_node_id = neighbor_fabric_node_id.value(),
                        .link_idx = link,
                    },
                };
            }

            spec.kernels.push_back(std::move(reader_spec));
            spec.kernels.push_back(std::move(writer_spec));
            spec.work_units.push_back(WorkUnitSpec{
                .name = "core_" + std::to_string(core_idx),
                .kernels = {reader_kernel, writer_kernel},
                .target_nodes = core,
            });
            run_args.kernel_run_args.push_back(std::move(reader_run_args));
            run_args.kernel_run_args.push_back(std::move(writer_run_args));
        }
        stick_start_id += num_sticks_to_read;
    }

    spec.advanced_options.fabric_connections = {std::move(fabric_connection)};

    return {
        .range = ttnn::MeshCoordinateRange(mesh_coord),
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts SliceReshardAsyncProgramFactory::create_mesh_workload_artifacts(
    const SliceReshardAsyncParams& args,
    const Tensor& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    const auto coords = tensor_coords.coords();
    artifacts.programs.reserve(coords.size());
    // Ring index and forward/backward neighbors depend on the coordinate, so each coord gets its own program.
    for (const auto& coord : coords) {
        artifacts.programs.push_back(
            CMAKE_UNIQUE_NAMESPACE::build_slice_reshard_program(args, coord, tensor_args, tensor_return_value));
    }
    return artifacts;
}

tt::tt_metal::ProgramRunArgs SliceReshardAsyncProgramFactory::override_runtime_arguments(
    const SliceReshardAsyncParams& args,
    const Tensor& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRange& /*range*/) {
    // final_semaphore / barrier_semaphore are excluded from
    // SliceReshardAsyncDeviceOperation::compute_program_hash, so they are re-applied here on every hit.
    return CMAKE_UNIQUE_NAMESPACE::make_dispatch_run_args(args, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim
