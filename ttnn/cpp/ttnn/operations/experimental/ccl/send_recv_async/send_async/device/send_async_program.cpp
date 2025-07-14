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
#include <ttnn/tensor/tensor_accessor_args.hpp>

using namespace tt::constants;

namespace ttnn {

tt::tt_metal::operation::ProgramWithCallbacks send_async_multicore(
    const Tensor& input_tensor,
    tt::tt_metal::IDevice* target_device,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;
    CoreCoord sender_core_coord;
    CoreCoord receiver_core_coord;
    tt::tt_fabric::FabricNodeId sender_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId receiver_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    for (const auto& connection : socket_connection_config) {
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            sender_core_coord = connection.sender_core.core_coord;
            receiver_core_coord = connection.receiver_core.core_coord;
            sender_fabric_node_id =
                input_tensor.mesh_device()->get_device_fabric_node_id(connection.sender_core.device_coord);
            receiver_fabric_node_id = mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord);
            break;
        }
    }
    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());
    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto num_pages = input_tensor.buffer()->num_pages();
    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;
    uint32_t num_whole_packets = 0, num_pages_remainder = 0, num_whole_packets_per_page = 0, partial_packet_size = 0,
             aligned_partial_packet_size = 0, socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        num_whole_packets = num_pages / num_pages_per_packet;
        num_pages_remainder = num_pages % num_pages_per_packet;
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
        aligned_partial_packet_size = tt::align(partial_packet_size, max_alignment);
        socket_block_size = socket_aligned_page_size;
    }

    uint32_t cb_num_pages = 2;
    uint32_t cb_page_size = fabric_max_payload_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    tt::tt_metal::CBHandle cb_src0_worker = CreateCircularBuffer(program, sender_core_coord, cb_src0_config);

    uint32_t packet_header_cb_num_pages = 2;  // One for data, one for sync
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    tt::tt_metal::CBHandle cb_packet_header_worker =
        CreateCircularBuffer(program, sender_core_coord, cb_packet_header_config);

    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer());
    auto compile_time_args = input_accessor_args.get_compile_time_args();
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,               // cb0_id
        num_pages,                   // num_pages
        input_page_size,             // input_page_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets,           // num_whole_packets
        num_pages_remainder,         // num_pages_remainder
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // fabric_max_payload_size
    };
    reader_compile_args.insert(reader_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    auto worker_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_reader.cpp",
        sender_core_coord,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    std::vector<uint32_t> reader_rt_args = {input_tensor.buffer()->address()};
    tt::tt_metal::SetRuntimeArgs(program, worker_reader_kernel_id, sender_core_coord, reader_rt_args);

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,               // cb0_id
        packet_header_cb_index,      // fabric_packet_header_cb_id
        num_pages,                   // num_pages
        socket_block_size,           // socket_block_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets,           // num_whole_packets
        num_pages_remainder,         // num_pages_remainder
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // fabric_max_payload_size
        socket_storage_in_dram,      // is_dram
    };

    auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_writer.cpp",
        sender_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
    uint32_t bank_id = 0;
    if (!socket_storage_in_dram) {
        bank_id = target_device->allocator()->get_bank_ids_from_logical_core(
            mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
    }

    std::vector<uint32_t> writer_rt_args = {mesh_socket.get_config_buffer()->address(), bank_id};

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);
    TT_FATAL(
        link_indices.size() > 0,
        "No link indices found {} {} {} {}",
        sender_fabric_node_id.mesh_id,
        sender_fabric_node_id.chip_id,
        receiver_fabric_node_id.mesh_id,
        receiver_fabric_node_id.chip_id);
    tt::tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id, receiver_fabric_node_id, link_indices[0], program, sender_core_coord, writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord, writer_rt_args);

    auto override_runtime_arguments_callback =
        [sender_core_coord, worker_reader_kernel_id, worker_writer_kernel_id](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& mesh_socket = static_cast<const ttnn::SendAsync*>(operation)->mesh_socket;
            auto& reader_runtime_args = GetRuntimeArgs(program, worker_reader_kernel_id, sender_core_coord);
            auto& writer_runtime_args = GetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord);

            reader_runtime_args[0] = input_tensors[0].buffer()->address();
            writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
