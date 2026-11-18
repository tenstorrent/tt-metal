// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_llama_sharded_program_factory.hpp"

#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace ttnn {

using namespace ccl;
using namespace tt::constants;

namespace experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// The names the per-coordinate programs and the cache-hit override share.
struct LlamaShardedSpecNames {
    tt::tt_metal::DFBSpecName data_dfb{"data"};
    tt::tt_metal::DFBSpecName packet_header_dfb{"packet_header"};
    tt::tt_metal::TensorParamName input_param{"input"};
    tt::tt_metal::TensorParamName output_param{"output"};
    tt::tt_metal::experimental::GlobalSemaphoreParamName semaphore_param{"semaphore"};
    tt::tt_metal::experimental::GlobalSemaphoreParamName barrier_semaphore_param{"barrier_semaphore"};
    tt::tt_metal::experimental::FabricConnectionSpecName forward_connection{"forward"};
    tt::tt_metal::experimental::FabricConnectionSpecName backward_connection{"backward"};
    tt::tt_metal::KernelSpecName reader_kernel{"reader"};
    tt::tt_metal::KernelSpecName writer_kernel{"writer"};
};

// The writer waits on and resets the barrier semaphore only when the caller supplies one and the output is not
// persistent.
bool uses_barrier_semaphore(const AllGatherAsyncParams& operation_attributes) {
    return operation_attributes.barrier_semaphore.has_value() && !operation_attributes.using_persistent_buffers;
}

// The run args every dispatch supplies: the io tensors and the caller-supplied GlobalSemaphores.
tt::tt_metal::ProgramRunArgs make_dispatch_run_args(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    const Tensor& output_tensor) {
    const LlamaShardedSpecNames names;
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.tensor_args.emplace(names.input_param, tensor_args.input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(names.output_param, output_tensor.mesh_tensor());
    run_args.advanced_options.global_semaphore_args.emplace(
        names.semaphore_param, std::cref(operation_attributes.semaphore.at(0)));
    if (uses_barrier_semaphore(operation_attributes)) {
        run_args.advanced_options.global_semaphore_args.emplace(
            names.barrier_semaphore_param, std::cref(operation_attributes.barrier_semaphore.value()));
    }
    return run_args;
}

// Pads each node's varargs with zeros to the longest list, so every node has the same vararg count.
uint32_t pad_varargs_to_common_length(
    tt::tt_metal::Table<tt::tt_metal::NodeCoord, tt::tt_metal::experimental::AdvancedKernelRunArgs::Varargs>& varargs) {
    size_t num_varargs = 0;
    for (const auto& [node, values] : varargs) {
        num_varargs = std::max(num_varargs, values.size());
    }
    for (auto& [node, values] : varargs) {
        values.resize(num_varargs, 0u);
    }
    return static_cast<uint32_t>(num_varargs);
}

ttnn::device_operation::MeshWorkloadArtifacts::PerCoordProgram build_llama_sharded_program(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherAsyncInputs& tensor_args,
    const Tensor& output_tensor) {
    using namespace tt::tt_metal;
    const LlamaShardedSpecNames names;
    const auto& input_tensor = tensor_args.input_tensor;

    const auto& sender_device_coord = mesh_coordinate;  // coord
    const auto& forward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto& backward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& ring_index = get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);  // device_index
    const auto& topology = operation_attributes.topology;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    bool use_optimal_ccl_for_llama = operation_attributes.use_optimal_ccl_for_llama;
    const bool use_barrier_sem = uses_barrier_semaphore(operation_attributes);

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_async_llama_sharded is called");

    ProgramSpec spec{.name = "all_gather_async_llama_sharded"};

    auto* mesh_device = input_tensor.device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }

    const bool enable_async_output_tensor = false;

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward] =
        get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);
    auto [forward_args, backward_args] = get_forward_backward_line_mcast_configuration(
        sender_device_coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        use_optimal_ccl_for_llama
            ? llama_specific::get_custom_worker_core_placement(num_links * num_workers_per_link)
            : ttnn::ccl::choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Llama sharded all-gather input buffer must be allocated on device");
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "Llama sharded all-gather output buffer must be allocated on device");

    // Tensor Info
    const auto input_tensor_num_pages = input_buffer->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch DFB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    uint32_t l1_scratch_dfb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_dfb_page_size_bytes;
    uint32_t dfb_num_pages =
        (input_tensor_num_pages / num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the DFB
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = names.data_dfb,
        .entry_size = l1_scratch_dfb_page_size_bytes,
        .num_entries = dfb_num_pages,
        .data_format_metadata = df,
    });
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = names.packet_header_dfb,
        .entry_size = static_cast<uint32_t>(packet_header_size_bytes),
        .num_entries = num_packet_headers_storable * 2,
        .data_format_metadata = tt::DataFormat::RawUInt32,
    });

    spec.tensor_parameters = {
        TensorParameter{.unique_id = names.input_param, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = names.output_param, .spec = output_tensor.tensor_spec()},
    };
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = names.semaphore_param, .memory_type = BufferType::L1},
    };
    if (use_barrier_sem) {
        spec.advanced_options.global_semaphore_parameters.push_back(
            {.unique_id = names.barrier_semaphore_param, .memory_type = BufferType::L1});
    }

    // KERNEL CREATION
    // Reader
    KernelSpec reader_kernel_spec{
        .unique_id = names.reader_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
                                        "llama_shapes_sharded_reader.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.data_dfb,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.input_param,
                    .accessor_name = "input",
                },
            },
        .compile_time_args =
            {
                {"my_chip_id", ring_index},
                {"tensor0_page_size", op_config.get_page_size()},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "num_tiles_per_core",
                        "num_tiles_to_read",
                        "first_core_tile_start_offset",
                        "num_cores",
                    },
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // Writer
    KernelSpec writer_kernel_spec{
        .unique_id = names.writer_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
                                        "llama_shapes_sharded_writer.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.data_dfb,
                    .accessor_name = "data",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // Only the writer touches the packet headers, so it is both producer and consumer.
                DFBBinding{
                    .dfb_spec_name = names.packet_header_dfb,
                    .accessor_name = "packet_header",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = names.packet_header_dfb,
                    .accessor_name = "packet_header",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = names.output_param,
                    .accessor_name = "output",
                },
            },
        .compile_time_args =
            {
                {"my_chip_id", ring_index},
                {"num_packet_headers_storable", num_packet_headers_storable},
                {"packet_size_in_pages", num_pages_per_packet},
                {"tensor0_page_size", op_config.get_page_size()},
                {"num_targets_forward_direction", num_targets_forward},
                {"num_targets_backward_direction", num_targets_backward},
                {"ring_size", ring_size},
                {"forward_mcast_dst_mesh_id", forward_args[0]},
                {"forward_mcast_dst_chip_id", forward_args[1]},
                {"forward_mcast_e_num_hops", forward_args[2]},
                {"forward_mcast_w_num_hops", forward_args[3]},
                {"forward_mcast_n_num_hops", forward_args[4]},
                {"forward_mcast_s_num_hops", forward_args[5]},
                {"backward_mcast_dst_mesh_id", backward_args[0]},
                {"backward_mcast_dst_chip_id", backward_args[1]},
                {"backward_mcast_e_num_hops", backward_args[2]},
                {"backward_mcast_w_num_hops", backward_args[3]},
                {"backward_mcast_n_num_hops", backward_args[4]},
                {"backward_mcast_s_num_hops", backward_args[5]},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "num_tiles_per_core",
                        "num_tiles_to_read",
                        "first_core_tile_start_offset",
                        "num_cores",
                        "wait_output_semaphore",
                        "reset_global_semaphore",
                        "out_ready_sem_noc0_x",
                        "out_ready_sem_noc0_y",
                        "out_ready_sem_wait_value",
                        "barrier_sem_noc0_x",
                        "barrier_sem_noc0_y",
                    },
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (use_barrier_sem) {
        writer_kernel_spec.compiler_options.defines.emplace("USE_BARRIER_SEM", "1");
    }
    writer_kernel_spec.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = names.semaphore_param,
            .accessor_name = "out_ready_semaphore",
        },
    };
    if (use_barrier_sem) {
        writer_kernel_spec.advanced_options.global_semaphore_bindings.push_back({
            .global_semaphore_parameter_name = names.barrier_semaphore_param,
            .accessor_name = "barrier_semaphore",
        });
    }
    writer_kernel_spec.advanced_options.fabric_connection_bindings = {
        {
            .fabric_connection_spec_name = names.forward_connection,
            .accessor_name = "forward",
        },
        {
            .fabric_connection_spec_name = names.backward_connection,
            .accessor_name = "backward",
        },
    };

    tt::tt_metal::experimental::FabricConnectionSpec forward_fabric_connection{
        .unique_id = names.forward_connection,
        .src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord),
    };
    tt::tt_metal::experimental::FabricConnectionSpec backward_fabric_connection{
        .unique_id = names.backward_connection,
        .src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord),
    };

    // Kernel Runtime Args
    KernelRunArgs reader_run_args{.kernel = names.reader_kernel};
    KernelRunArgs writer_run_args{.kernel = names.writer_kernel};
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - (1 / ring_size);
    uint32_t start_core_index_for_device = output_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;
    TT_FATAL(
        output_cores_vec.size() % ring_size == 0 || output_cores_vec.size() == 1,
        "output sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution scheme",
        output_cores_vec.size(),
        ring_size);
    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);
    log_trace(tt::LogOp, "output_cores_this_device: {}", output_cores_this_device);
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t output_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % output_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / output_tensor_shard_num_pages;
             i < (input_tile_id_end + output_tensor_shard_num_pages - 1) / output_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }
        // Set reader runtime args
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", input_tensor_shard_num_pages},
                {"num_tiles_to_read", worker_num_tiles_to_read},
                {"first_core_tile_start_offset", input_first_core_tile_start_offset},
                {"num_cores", static_cast<uint32_t>(input_tensor_cores_x.size())},
            });
        // The input cores' x coordinates, then their y coordinates.
        auto& reader_varargs = reader_run_args.advanced_options.runtime_varargs[core];
        reader_varargs.insert(reader_varargs.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_varargs.insert(reader_varargs.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles_per_core", output_tensor_shard_num_pages},
                {"num_tiles_to_read", worker_num_tiles_to_read},
                {"first_core_tile_start_offset", output_first_core_tile_start_offset},
                {"num_cores", static_cast<uint32_t>(output_tensor_cores_x.size())},
                {"wait_output_semaphore", static_cast<uint32_t>(wait_output_semaphore)},
                {"reset_global_semaphore", static_cast<uint32_t>(reset_global_semaphore)},
                {"out_ready_sem_noc0_x", static_cast<uint32_t>(drain_sync_core.x)},
                {"out_ready_sem_noc0_y", static_cast<uint32_t>(drain_sync_core.y)},
                {"out_ready_sem_wait_value", out_ready_sem_wait_value},
                {"barrier_sem_noc0_x", static_cast<uint32_t>(barrier_core.x)},
                {"barrier_sem_noc0_y", static_cast<uint32_t>(barrier_core.y)},
            });
        // The output cores' x coordinates, then their y coordinates.
        auto& writer_varargs = writer_run_args.advanced_options.runtime_varargs[core];
        writer_varargs.insert(writer_varargs.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_varargs.insert(writer_varargs.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());

        if (forward_coord.has_value()) {
            forward_fabric_connection.targets[core] = {
                tt::tt_metal::experimental::FabricConnectionTarget{
                    .dst_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value()),
                    .link_idx = link,
                },
            };
        }
        if (backward_coord.has_value()) {
            backward_fabric_connection.targets[core] = {
                tt::tt_metal::experimental::FabricConnectionTarget{
                    .dst_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value()),
                    .link_idx = link,
                },
            };
        }
    }
    reader_kernel_spec.advanced_options.num_runtime_varargs =
        pad_varargs_to_common_length(reader_run_args.advanced_options.runtime_varargs);
    writer_kernel_spec.advanced_options.num_runtime_varargs =
        pad_varargs_to_common_length(writer_run_args.advanced_options.runtime_varargs);

    spec.kernels = {
        std::move(reader_kernel_spec),
        std::move(writer_kernel_spec),
    };
    spec.work_units = {
        WorkUnitSpec{
            .name = "worker",
            .kernels = {names.reader_kernel, names.writer_kernel},
            .target_nodes = sender_worker_core_range,
        },
    };
    spec.advanced_options.fabric_connections = {
        std::move(forward_fabric_connection),
        std::move(backward_fabric_connection),
    };

    ProgramRunArgs run_args = make_dispatch_run_args(operation_attributes, tensor_args, output_tensor);
    run_args.kernel_run_args = {
        std::move(reader_run_args),
        std::move(writer_run_args),
    };

    return {
        .range = ttnn::MeshCoordinateRange(mesh_coordinate),
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts LlamaShardedMeshWorkloadFactory::create_mesh_workload_artifacts(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    const auto coords = tensor_coords.coords();
    artifacts.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        artifacts.programs.push_back(CMAKE_UNIQUE_NAMESPACE::build_llama_sharded_program(
            operation_attributes, coord, tensor_args, output_tensor));
    }
    return artifacts;
}

tt::tt_metal::ProgramRunArgs LlamaShardedMeshWorkloadFactory::override_runtime_arguments(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRange& /*range*/) {
    // Caller-owned semaphores stay out of the program hash, so every dispatch supplies them again.
    return CMAKE_UNIQUE_NAMESPACE::make_dispatch_run_args(operation_attributes, tensor_args, output_tensor);
}

}  // namespace experimental::prim

}  // namespace ttnn
