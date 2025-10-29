// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

using namespace tt::constants;

namespace ttnn {

tt::tt_metal::operation::ProgramWithCallbacks send_async_multicore(
    const Tensor& input_tensor,
    tt::tt_metal::IDevice* target_device,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    // Collect all matching sender-receiver core pairs for this device
    std::vector<CoreCoord> sender_core_coords;
    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<size_t> connection_indices;  // Track original connection indices for proper device/bank lookup

    printf("len of socket_connection_config: %zu\n", socket_connection_config.size());
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
            printf("pushing conn_idx %zu to connection_indices for device id %d\n", conn_idx, target_device->id());
        }
    }

    TT_FATAL(!sender_core_coords.empty(), "No sender cores found for target device {}", target_device->id());
    uint32_t num_cores = sender_core_coords.size();
    printf("Found %u sender cores for device %d\n", num_cores, target_device->id());

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());
    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto total_num_pages = input_tensor.buffer()->num_pages();

    // Divide workload per device: same work per device, different work per core within device
    // Each device processes the full tensor, but cores within device divide the work
    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    // Calculate per-core parameters that will be used for each core
    uint32_t num_whole_packets_per_page = 0, partial_packet_size = 0, aligned_partial_packet_size = 0,
             socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
        socket_block_size = socket_aligned_page_size;
    }

    uint32_t cb_num_pages = 2;
    uint32_t cb_page_size = fabric_max_payload_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    // Create CoreRangeSet from sender cores
    std::set<CoreRange> sender_core_ranges;
    for (const auto& core : sender_core_coords) {
        sender_core_ranges.insert(CoreRange(core));
    }
    CoreRangeSet sender_core_range_set(sender_core_ranges);

    // Create circular buffers for all sender cores
    CreateCircularBuffer(program, sender_core_range_set, cb_src0_config);

    uint32_t packet_header_cb_num_pages = 2;  // One for data, one for sync
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, sender_core_range_set, cb_packet_header_config);

    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

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
    };
    reader_compile_args.insert(reader_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    // Create kernels once for all cores using CoreRangeSet
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_reader.cpp",
        sender_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                // cb0_id
        packet_header_cb_index,       // fabric_packet_header_cb_id
        socket_block_size,            // socket_block_size
        socket_aligned_page_size,     // socket_page_size
        num_pages_per_packet,         // num_pages_per_packet
        num_whole_packets_per_page,   // num_whole_packets_per_page
        aligned_partial_packet_size,  // aligned_partial_packet_size
        fabric_max_payload_size,      // whole_packet_size (fabric_max_payload_size)
        socket_storage_in_dram,       // is_dram
    };

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_writer.cpp",
        sender_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // Set runtime arguments for each core individually
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& sender_core_coord = sender_core_coords[core_idx];
        const auto& receiver_core_coord = receiver_core_coords[core_idx];

        // Calculate pages for this core using local core index within device
        uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

        // Calculate cumulative start offset: sum of pages assigned to all previous cores within this device
        uint32_t page_start_offset = 0;
        for (uint32_t prev_idx = 0; prev_idx < core_idx; ++prev_idx) {
            uint32_t prev_pages = pages_per_core + (prev_idx < remainder_pages ? 1 : 0);
            page_start_offset += prev_pages;
        }

        // Calculate per-core packet parameters
        uint32_t num_whole_packets = 0, num_pages_remainder = 0;
        if (num_pages_per_packet > 0) {
            num_whole_packets = pages_for_this_core / num_pages_per_packet;
            num_pages_remainder = pages_for_this_core % num_pages_per_packet;
        }

        // Runtime arguments for reader (vary per core)
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // input_base_addr
            pages_for_this_core,               // num_pages (for this core)
            page_start_offset,                 // page_start_offset
            num_whole_packets,                 // num_whole_packets (for this core)
            num_pages_remainder,               // num_pages_remainder (for this core)
        };
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core_coord, reader_rt_args);

        // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
        uint32_t bank_id = 0;
        if (!socket_storage_in_dram) {
            // Get the receiver device to determine the correct bank ID for the receiver core
            const auto& connection = socket_connection_config[connection_indices[core_idx]];
            auto* receiver_device = socket_mesh_device->get_device(connection.receiver_core.device_coord);
            bank_id = receiver_device->allocator()->get_bank_ids_from_logical_core(
                mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
        }

        // Runtime arguments for writer (vary per core)
        std::vector<uint32_t> writer_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            bank_id,                                     // bank_id
            pages_for_this_core,                         // num_pages (for this core)
            page_start_offset,                           // page_start_offset (for this core)
            num_whole_packets,                           // num_whole_packets (for this core)
            num_pages_remainder,                         // num_pages_remainder (for this core)
        };

        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];
        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);
        TT_FATAL(
            !link_indices.empty(),
            "No link indices found {} {} {} {}",
            sender_fabric_node_id.mesh_id,
            sender_fabric_node_id.chip_id,
            receiver_fabric_node_id.mesh_id,
            receiver_fabric_node_id.chip_id);

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        printf(
            "SEND CORE %u: pages=%u, start_offset=%u, link=%u, whole_packets=%u, remainder=%u\n",
            core_idx,
            pages_for_this_core,
            page_start_offset,
            selected_link_index,
            num_whole_packets,
            num_pages_remainder);
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            selected_link_index,
            program,
            sender_core_coord,
            writer_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core_coord, writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [sender_core_coords, reader_kernel_id, writer_kernel_id](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& mesh_socket = static_cast<const ttnn::SendAsync*>(operation)->mesh_socket;

            // Update runtime args for all cores
            for (uint32_t core_idx = 0; core_idx < sender_core_coords.size(); ++core_idx) {
                const auto& sender_core_coord = sender_core_coords[core_idx];
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, sender_core_coord);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, sender_core_coord);

                // Update only the tensor buffer address, keep other runtime args unchanged
                reader_runtime_args[0] = input_tensors[0].buffer()->address();
                writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
                // Note: pages_for_this_core, page_start_offset, num_whole_packets, num_pages_remainder
                // remain the same as set during kernel creation
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
