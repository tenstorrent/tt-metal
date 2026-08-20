// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_send_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"
using namespace tt::constants;

namespace ttnn::experimental::prim {

BufferedSendMeshWorkloadFactory::cached_mesh_workload_t BufferedSendMeshWorkloadFactory::create_mesh_workload(
    const BufferedSendParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    ttnn::MeshCoordinateRangeSet workload_coords =
        ttnn::send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
            tensor_coords, operation_attributes.mesh_socket);

    for (const auto& coord : workload_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<BufferedSendMeshWorkloadFactory::shared_variables_t>
BufferedSendMeshWorkloadFactory::create_at(
    const BufferedSendParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    auto mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    auto* mesh_device = input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> sender_core_coords;
    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<size_t> connection_indices;

    for (size_t conn_idx = 0; conn_idx < socket_connection_config.size(); ++conn_idx) {
        const auto& connection = socket_connection_config[conn_idx];
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            sender_core_coords.push_back(connection.sender_core.core_coord);
            receiver_core_coords.push_back(connection.receiver_core.core_coord);
            sender_fabric_node_ids.push_back(
                input_tensor.device()->get_fabric_node_id(connection.sender_core.device_coord));
            receiver_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord));
            connection_indices.push_back(conn_idx);
        }
    }
    uint32_t num_cores = sender_core_coords.size();

    // cores must not exceed available fabric links
    if (num_cores > 0) {
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[0];
        const auto& sender_fabric_node_id = sender_fabric_node_ids[0];
        auto available_link_indices =
            tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
        uint32_t num_available_links = available_link_indices.size();

        TT_FATAL(
            num_cores <= num_available_links,
            "Cannot create {} receiver-sender pairs with only {} available fabric links between devices. "
            "Reduce the number of cores per device. "
            "Available links: {}, Requested pairs: {}",
            num_cores,
            num_available_links,
            num_available_links,
            num_cores);
    }

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());
    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto total_num_pages = input_tensor.buffer()->num_pages();

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    // Bank-contiguous packing optimization (see send_direct_async): for an interleaved DRAM tensor,
    // pages whose indices differ by num_banks live in the same bank at consecutive slots and are
    // contiguous in memory. Each CB entry holds a super-block of (num_banks * num_pages_per_packet)
    // pages; the reader issues one read per bank (overlapping across banks before a single barrier)
    // and the writer drains the entry as num_banks combined fabric packets. Restricted to DRAM so the
    // super-block CB stays small (num_banks ~12).
    auto* input_buffer = input_tensor.buffer();
    const bool is_interleaved = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
    const bool is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    uint32_t num_banks = input_buffer->allocator()->get_num_banks(input_buffer->buffer_type());
    uint32_t enable_bank_packing = (is_interleaved && is_dram && num_pages_per_packet > 1 && num_banks > 1) ? 1u : 0u;

    // Small pages (num_pages_per_packet > 0): pack multiple whole pages into one fabric packet.
    // Large pages (num_pages_per_packet == 0): split a single page across multiple fabric packets.
    uint32_t num_whole_packets_per_page = 0, partial_packet_size = 0;
    if (num_pages_per_packet == 0) {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
    }

    // Handshake page carries the sender buffer address (advertise) and the completion token. Size it
    // to comfortably hold the dest-info struct and keep it independent of the data page size.
    uint32_t handshake_page_size = tt::align(static_cast<uint32_t>(64), max_alignment);

    uint32_t cb_num_pages = 2;
    // For bank packing, each CB entry holds a full super-block: num_banks regions of
    // num_pages_per_packet pages each (laid out at input_page_size stride, contiguous per bank).
    uint32_t cb_page_size = enable_bank_packing
                                ? static_cast<uint32_t>(num_banks * num_pages_per_packet * input_page_size)
                                : fabric_max_payload_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    std::set<CoreRange> sender_core_ranges;
    for (const auto& core : sender_core_coords) {
        sender_core_ranges.insert(CoreRange(core));
    }
    CoreRangeSet sender_core_range_set(sender_core_ranges);

    CreateCircularBuffer(program, sender_core_range_set, cb_src0_config);

    uint32_t packet_header_cb_num_pages = 2;  // One for data, one for sync
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, sender_core_range_set, cb_packet_header_config);

    // Persistent L1_SMALL buffer backing the handshake: page 0 is the dest-info landing zone (the
    // receiver writes the OutputTensorInfo struct back here), page 1 stages the advertise payload
    // pushed to the receiver over the socket. Unlike a CB, this allocation persists across program
    // reuse and is not aliased by the data CBs, which the buffered ring relies on. Sharded one page
    // per sender core (HEIGHT_SHARDED) so every sender core sees the struct at the same L1 address,
    // mirroring how GlobalSemaphore allocates its persistent L1_SMALL buffer.
    uint32_t info_buffer_page_size = 256 * sizeof(uint32_t);
    auto info_shard_parameters = tt::tt_metal::ShardSpecBuffer(
        sender_core_range_set, {1, 1}, tt::tt_metal::ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    tt::tt_metal::ShardedBufferConfig info_buffer_config = {
        .device = input_tensor.device(),
        .size = num_cores * info_buffer_page_size,
        .page_size = info_buffer_page_size,
        .buffer_type = tt::tt_metal::BufferType::L1_SMALL,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(info_shard_parameters),
    };
    auto info_buffer = tt::tt_metal::distributed::AnyBuffer::create(info_buffer_config);
    auto info_buffer_addr = static_cast<uint32_t>(info_buffer.get_buffer()->address());

    log_info(tt::LogOp, "Writing zeros to info buffer");
    // Zero-initialize the info buffer (blocking, so the reset lands before the program runs).
    std::vector<uint32_t> zeros(256 * num_cores, 0);
    auto info_mesh_buffer = info_buffer.get_mesh_buffer();
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        mesh_device->mesh_command_queue(), info_mesh_buffer, zeros, /*blocking=*/true);

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer());
    auto compile_time_args = input_accessor_args.get_compile_time_args();
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,               // cb0_id
        input_page_size,             // input_page_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // fabric_max_payload_size
        num_banks,                   // num_banks
        enable_bank_packing,         // enable_bank_packing
    };
    reader_compile_args.insert(reader_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    // Reuse send_direct_async's reader: it streams input pages into the data CB, gathering
    // bank-contiguous pages into super-block CB entries when bank packing is enabled.
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/device/kernels/"
        "sender_direct_reader.cpp",
        sender_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // The writer addresses the receiver's output tensor, whose layout matches the input tensor.
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,               // cb0_id
        packet_header_cb_index,      // fabric_packet_header_cb_id
        handshake_page_size,         // handshake_page_size (socket page size)
        input_page_size,             // output_page_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // whole_packet_size (fabric_max_payload_size)
        num_banks,                   // num_banks
        enable_bank_packing,         // enable_bank_packing
    };
    writer_compile_args.insert(writer_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    // buffered_send uses its own writer kernel: it receives the ring of receive-buffer addresses
    // advertised by buffered_recv and streams the tensor into the selected free buffer.
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/buffered_send/device/kernels/"
        "sender_buffered.cpp",
        sender_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& sender_core_coord = sender_core_coords[core_idx];
        uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

        uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
        uint32_t num_whole_packets = 0, num_pages_remainder = 0;
        if (num_pages_per_packet > 0) {
            num_whole_packets = pages_for_this_core / num_pages_per_packet;
            num_pages_remainder = pages_for_this_core % num_pages_per_packet;
        }

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // input_base_addr
            pages_for_this_core,               // num_pages
            page_start_offset,                 // page_start_offset
            num_whole_packets,                 // num_whole_packets
            num_pages_remainder,               // num_pages_remainder
        };
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core_coord, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            pages_for_this_core,                         // num_pages
            page_start_offset,                           // page_start_offset
            num_whole_packets,                           // num_whole_packets
            num_pages_remainder,                         // num_pages_remainder
            info_buffer_addr,                            // handshake_info_buffer_addr (persistent L1_SMALL)
        };

        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];
        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            selected_link_index,
            program,
            sender_core_coord,
            writer_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core_coord, writer_rt_args);
    }

    return {
        std::move(program),
        shared_variables_t{
            .sender_core_coords = sender_core_coords,
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .info_buffer = std::move(info_buffer),
        }};
}

void BufferedSendMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const BufferedSendParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& sender_core_coords = shared_vars.sender_core_coords;
        const auto& reader_kernel_id = shared_vars.reader_kernel_id;
        const auto& writer_kernel_id = shared_vars.writer_kernel_id;

        const auto& mesh_socket = operation_attributes.mesh_socket;
        const auto& input_tensor = tensor_args;

        for (const auto& sender_core_coord : sender_core_coords) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, sender_core_coord);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, sender_core_coord);

            reader_runtime_args[0] = input_tensor.buffer()->address();
            writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
            // The persistent L1_SMALL buffer keeps a stable address across reuse, but refresh it
            // defensively in case the cached allocation was rebuilt.
            writer_runtime_args[5] = static_cast<uint32_t>(shared_vars.info_buffer.get_buffer()->address());
        }
    }
}

}  // namespace ttnn::experimental::prim
