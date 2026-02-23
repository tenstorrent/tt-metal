// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_via_broadcast_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn {

using namespace ccl;

namespace experimental::prim {

AllGatherViaBroadcastFactory::cached_mesh_workload_t AllGatherViaBroadcastFactory::create_mesh_workload(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
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
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args.input_tensor,
            output_tensor,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherViaBroadcastFactory::cached_program_t AllGatherViaBroadcastFactory::create_at(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::Program program{};

    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        ring_size, ring_index, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    auto* mesh_device = input_tensor.device();
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] = ::ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt);

    const uint32_t MAX_PACKET_SIZE_BYTES = std::bit_floor(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();
    uint32_t cb_page_size = std::lcm(std::lcm(input_page_size, output_page_size), MAX_PACKET_SIZE_BYTES);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // 32^2 elements == 1/2 or 1 packet, a couple more packets per cb_page for less sync
        cb_page_size *= 4;
    }

    // per device input and output page numbers
    uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    for (uint32_t i = 0; i < operation_attributes.dim; ++i) {
        num_input_pages /= input_tensor.logical_shape()[i];
    }
    uint32_t num_output_pages = (num_input_pages * input_page_size) / output_page_size;
    // offset into the gathered tensor
    uint32_t write_page_offset = num_output_pages * ring_index;

    // L1 Scratch CB Creation
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(3 * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,    // cb0_id
        input_page_size,  // page_size
        cb_page_size,     // page_size
    };

    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,  // cb0_id
        cb_page_size,
        output_page_size,
        MAX_PACKET_SIZE_BYTES,  // packet_size,
        num_targets_forward,    // num_targets_forward_direction
        num_targets_backward,   // num_targets_backward_direction
    };

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
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t input_pages_per_link = num_input_pages / operation_attributes.num_links;
        uint32_t remainder = num_input_pages % operation_attributes.num_links;
        uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            input_tile_id_start,               // tile_id_start
            input_tile_id_end,                 // tile_id_end
        };

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = ring_size * operation_attributes.num_links;

        uint32_t output_tile_id_start = (input_tile_id_start * num_output_pages) / num_input_pages;
        uint32_t output_tile_id_end = (input_tile_id_end * num_output_pages) / num_input_pages;
        // page id in gathered tensor with the write page offset
        output_tile_id_start += write_page_offset;
        output_tile_id_end += write_page_offset;
        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // tensor_address0  //HERE
            semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
            barrier_semaphore.address(),        // barrier_sem
            output_tile_id_start,
            output_tile_id_end,
            wait_output_semaphore,     // wait_output_semaphore
            reset_global_semaphore,    // reset_global_semaphore
            drain_sync_core.x,         // out_ready_sem_noc0_x
            drain_sync_core.y,         // out_ready_sem_noc0_y
            out_ready_sem_wait_value,  // out_ready_sem_wait_value
            barrier_core.x,            // barrier_sem_noc0_x
            barrier_core.y             // barrier_sem_noc0_y
        };
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);

        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
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

void AllGatherViaBroadcastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherAsyncParams& /*operation_attributes*/,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
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
            worker_reader_sender_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = output_tensor.buffer()->address();
            worker_writer_sender_runtime_args[1] = shared_vars.semaphore.address();
            worker_writer_sender_runtime_args[2] = shared_vars.barrier_semaphore.address();
        }
    }
}

}  // namespace experimental::prim
}  // namespace ttnn
