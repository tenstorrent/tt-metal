// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_minimal_broadcast::program {

DeepseekMinimalBroadcastProgramFactory::cached_mesh_workload_t
DeepseekMinimalBroadcastProgramFactory::create_mesh_workload(
    const DeepseekMinimalBroadcastParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DeepseekMinimalBroadcastInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = DeepseekMinimalBroadcastProgramFactory::create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

DeepseekMinimalBroadcastProgramFactory::cached_program_t DeepseekMinimalBroadcastProgramFactory::create_at(
    const DeepseekMinimalBroadcastParams& operation_attributes,
    const MeshCoordinate& self_coord,
    const DeepseekMinimalBroadcastInputs& tensor_args,
    Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};

    uint32_t num_links = operation_attributes.num_links;
    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, self_coord, operation_attributes.cluster_axis);
    MeshCoordinate sender_coord = operation_attributes.sender_coord;
    auto topology = operation_attributes.topology;
    auto cluster_axis = operation_attributes.cluster_axis;

    std::optional<MeshCoordinate> forward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, 1, topology, cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, -1, topology, cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        self_coord,
        is_first_chip,
        is_last_chip);

    // Fatal if input is not tilized with tiny tiles (1, 32)
    bool tilized = input_tensor.layout() == ttnn::TILE_LAYOUT;
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    TT_FATAL(
        tilized,
        "broadcast_batch1 op currently only supports TILE_LAYOUT input tensors. Got layout: {}",
        input_tensor.layout());
    TT_FATAL(
        tile_width == 32 && tile_height == 1,
        "broadcast_batch1 op currently only supports TILE_LAYOUT input tensors with tile size (1, 32). Got tile size: "
        "({}, {})",
        tile_height,
        tile_width);

    // Extract shard grid from input tensor
    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid
    std::vector<CoreCoord> cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto c = corerange_to_cores(core_range, std::nullopt);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    // data should be on a single core: Fatal if not
    TT_FATAL(cores.size() == 1, "Input tensor must be sharded to a single core");

    // worker core should be the data core if only one link is used
    // if 2 links are used, worker cores should be the data core and another core next to it
    // next to it could be any available core at x-1 or x+1
    std::vector<CoreCoord> sender_worker_cores;
    sender_worker_cores.push_back(cores[0]);
    if (num_links == 2) {
        CoreCoord next_core;
        if (cores[0].x > 0) {
            next_core = CoreCoord{cores[0].x - 1, cores[0].y};
        } else {
            next_core = CoreCoord{cores[0].x + 1, cores[0].y};
        }
        sender_worker_cores.push_back(next_core);
    }
    auto sender_worker_core_range = CoreRangeSet(sender_worker_cores);

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] =
        ::ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    // Tensor Info - get num_pages early for CB config
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();

    uint32_t l1_scratch_cb_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    const auto tiny_tile = tt::tt_metal::Tile({1, 32});

    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    // CB page size should be individual tile size, not the entire packet
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_pages_per_packet * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes)
            .set_tile_dims(src0_cb_index, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Sender info
    bool is_sender = self_coord == sender_coord;

    auto data_core_coord = mesh_device->worker_core_from_logical_core(sender_worker_cores[0]);
    auto core_noc_x = data_core_coord.x;
    auto core_noc_y = data_core_coord.y;

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        is_sender,
        core_noc_x,
        core_noc_y};

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        num_targets_forward,                         // num_targets_forward_direction
        num_targets_backward,                        // num_targets_backward_direction
        is_sender,
        core_noc_x,
        core_noc_y};

    std::vector<uint32_t> mcast_forward_args(2, 0);
    std::vector<uint32_t> mcast_backward_args(2, 0);
    if (forward_coord.has_value()) {
        mcast_forward_args[0] = 1;
        mcast_forward_args[1] = num_targets_forward;
    }
    if (backward_coord.has_value()) {
        mcast_backward_args[0] = 1;
        mcast_backward_args[1] = num_targets_backward;
    }
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/kernels/"
        "broadcast_tile_reader_batch1.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/kernels/"
        "broadcast_tile_writer_batch1.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;

        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            input_tile_id_start,               // tile_id_start
            input_tile_id_end,                 // tile_id_end
        };
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !is_sender;
        bool reset_global_semaphore = (link == 0) && !is_sender;
        uint32_t out_ready_sem_wait_value = 1 * num_links;
        uint32_t output_tile_id_start = input_tile_id_start;
        uint32_t output_tile_id_end = input_tile_id_end;

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // tensor_address0  //HERE
            semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
            output_tile_id_start,               // tile_id_start
            output_tile_id_end,                 // tile_id_end
            wait_output_semaphore,              // wait_output_semaphore
            reset_global_semaphore,             // reset_global_semaphore
            drain_sync_core.x,                  // out_ready_sem_noc0_x
            drain_sync_core.y,                  // out_ready_sem_noc0_y
            out_ready_sem_wait_value,           // out_ready_sem_wait_value
            barrier_semaphore.address(),        // barrier_sem
            barrier_core.x,                     // barrier_sem_noc0_x
            barrier_core.y                      // barrier_sem_noc0_y
        };
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);

        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(self_coord);
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
        dst_nodes.reserve(num_connections);
        if (forward_coord.has_value()) {
            const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            dst_nodes.push_back(forward_coord_fabric_node_id);
        }
        if (backward_coord.has_value()) {
            const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            dst_nodes.push_back(backward_coord_fabric_node_id);
        }

        append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id, dst_nodes, {link}, program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = sender_worker_cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .semaphore = semaphore,
        .barrier_semaphore = barrier_semaphore,
        .ring_index = ring_index,
    };
    return {std::move(program), std::move(shared_variables)};
}

void DeepseekMinimalBroadcastProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DeepseekMinimalBroadcastParams& /*operation_attributes*/,
    const DeepseekMinimalBroadcastInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_return_value;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        // const auto& coord = coordinate_range.start_coord();
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        log_trace(tt::LogOp, "DEBUG: semaphore: {}", shared_vars.semaphore.address());
        log_trace(tt::LogOp, "DEBUG: barrier_semaphore: {}", shared_vars.barrier_semaphore.address());
        // update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);

        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();

            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = output.buffer()->address();
            worker_writer_sender_runtime_args[1] = shared_vars.semaphore.address();
            worker_writer_sender_runtime_args[9] = shared_vars.barrier_semaphore.address();
        }
    }
}
}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_broadcast::program
