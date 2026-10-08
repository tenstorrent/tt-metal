// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// The socket connections whose sender core sits on `target_device`, in socket-connection order.
// create_descriptor and override_runtime_arguments both walk this so the per-core runtime-arg
// ordering they assume stays identical.
struct SendAsyncConnections {
    std::vector<CoreCoord> sender_core_coords;
    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<size_t> connection_indices;
};

SendAsyncConnections collect_send_async_connections(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const Tensor& input_tensor,
    tt::tt_metal::IDevice* target_device) {
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    SendAsyncConnections connections;
    connections.sender_core_coords.reserve(socket_connection_config.size());
    connections.receiver_core_coords.reserve(socket_connection_config.size());
    connections.sender_fabric_node_ids.reserve(socket_connection_config.size());
    connections.receiver_fabric_node_ids.reserve(socket_connection_config.size());
    connections.connection_indices.reserve(socket_connection_config.size());

    for (size_t conn_idx = 0; conn_idx < socket_connection_config.size(); ++conn_idx) {
        const auto& connection = socket_connection_config[conn_idx];
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            connections.sender_core_coords.push_back(connection.sender_core.core_coord);
            connections.receiver_core_coords.push_back(connection.receiver_core.core_coord);
            connections.sender_fabric_node_ids.push_back(
                input_tensor.device()->get_fabric_node_id(connection.sender_core.device_coord));
            connections.receiver_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord));
            connections.connection_indices.push_back(conn_idx);
        }
    }
    return connections;
}

// Descriptor kernel indices, fixed by the push order at the end of create_descriptor.
constexpr uint32_t send_async_reader_kernel_index = 0;
constexpr uint32_t send_async_writer_kernel_index = 1;

// Runtime-arg slots re-applied by override_runtime_arguments.
constexpr uint32_t send_async_reader_input_addr_arg_index = 0;
constexpr uint32_t send_async_writer_socket_config_addr_arg_index = 0;

}  // namespace

tt::tt_metal::ProgramDescriptor SendAsyncProgramFactory::create_descriptor(
    const SendAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;

    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(input_tensor, mesh_dispatch_coordinate, "send_async");
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    auto connections = collect_send_async_connections(mesh_socket, input_tensor, target_device);
    const auto& sender_core_coords = connections.sender_core_coords;
    const auto& receiver_core_coords = connections.receiver_core_coords;
    const auto& sender_fabric_node_ids = connections.sender_fabric_node_ids;
    const auto& receiver_fabric_node_ids = connections.receiver_fabric_node_ids;
    const auto& connection_indices = connections.connection_indices;

    uint32_t num_cores = sender_core_coords.size();
    // This device holds no sender core of the socket, so it has no work. An empty descriptor tells
    // the framework to skip this coordinate.
    if (num_cores == 0) {
        return ProgramDescriptor{};
    }

    // cores must not exceed available fabric links
    {
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

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "send_async: input tensor buffer is null");

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_buffer->alignment());
    auto input_page_size = input_buffer->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto total_num_pages = input_buffer->num_pages();

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    uint32_t num_whole_packets_per_page = 0, partial_packet_size = 0, socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
        socket_block_size = socket_aligned_page_size;
    }

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    std::set<CoreRange> sender_core_ranges;
    for (const auto& core : sender_core_coords) {
        sender_core_ranges.insert(CoreRange(core));
    }
    CoreRangeSet sender_core_range_set(sender_core_ranges);

    ProgramDescriptor desc;

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb_num_pages = 2;
    uint32_t cb_page_size = fabric_max_payload_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * cb_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = df,
            .page_size = cb_page_size,
        }}},
    });

    constexpr uint8_t packet_header_cb_index = tt::CBIndex::c_1;
    uint32_t packet_header_cb_num_pages = 2;  // One for data, one for sync
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = packet_header_cb_index,
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size,
        }}},
    });

    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;
    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_buffer);
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

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = sender_core_range_set;
    reader.compile_time_args = std::move(reader_compile_args);
    reader.named_compile_time_args = {{"cb0_id", src0_cb_index}};
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,               // cb0_id
        packet_header_cb_index,      // fabric_packet_header_cb_id
        socket_block_size,           // socket_block_size
        socket_aligned_page_size,    // socket_page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // whole_packet_size (fabric_max_payload_size)
        socket_storage_in_dram,      // is_dram
    };

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = sender_core_range_set;
    writer.compile_time_args = std::move(writer_compile_args);
    writer.named_compile_time_args = {
        {"cb0_id", src0_cb_index},
        {"fabric_packet_header_cb_id", packet_header_cb_index},
    };
    writer.config = WriterConfigDescriptor{};

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& sender_core_coord = sender_core_coords[core_idx];
        const auto& receiver_core_coord = receiver_core_coords[core_idx];
        uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

        uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
        uint32_t num_whole_packets = 0, num_pages_remainder = 0;
        if (num_pages_per_packet > 0) {
            num_whole_packets = pages_for_this_core / num_pages_per_packet;
            num_pages_remainder = pages_for_this_core % num_pages_per_packet;
        }
        reader.emplace_runtime_args(
            sender_core_coord,
            {
                input_buffer,         // input_base_addr
                pages_for_this_core,  // num_pages
                page_start_offset,    // page_start_offset
                num_whole_packets,    // num_whole_packets
                num_pages_remainder,  // num_pages_remainder
            });

        // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
        uint32_t bank_id = 0;
        if (!socket_storage_in_dram) {
            const auto& connection = socket_connection_config[connection_indices[core_idx]];
            auto* receiver_device = socket_mesh_device->get_device(connection.receiver_core.device_coord);
            bank_id = receiver_device->allocator()->get_bank_ids_from_logical_core(
                mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
        } else {
            // Assign DRAM banks in round-robin for each receiver core
            auto num_dram_banks = target_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
            bank_id = core_idx % num_dram_banks;
        }

        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];
        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        std::vector<uint32_t> fabric_connection_rt_args;
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            selected_link_index,
            desc,
            sender_core_coord,
            fabric_connection_rt_args);

        // The socket config buffer is not tensor-backed and its address is not in the program hash, so
        // it cannot be a Buffer* binding; override_runtime_arguments re-applies it on every cache hit.
        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(socket_config_addr);   // smuggled-rta-ok: re-applied via override_runtime_arguments
        writer_rt_args.push_back(bank_id);              // bank_id
        writer_rt_args.push_back(pages_for_this_core);  // num_pages
        writer_rt_args.push_back(page_start_offset);    // page_start_offset
        writer_rt_args.push_back(num_whole_packets);    // num_whole_packets
        writer_rt_args.push_back(num_pages_remainder);  // num_pages_remainder
        writer_rt_args.append(fabric_connection_rt_args);
        writer.emplace_runtime_args(sender_core_coord, writer_rt_args);
    }

    TT_FATAL(desc.kernels.size() == send_async_reader_kernel_index, "send_async: reader kernel index mismatch");
    desc.kernels.push_back(std::move(reader));
    TT_FATAL(desc.kernels.size() == send_async_writer_kernel_index, "send_async: writer kernel index mismatch");
    desc.kernels.push_back(std::move(writer));

    return desc;
}

void SendAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SendAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    tt::tt_metal::IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(input_tensor, mesh_dispatch_coordinate, "send_async");

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "send_async: input tensor buffer is null");

    // Everything else in the runtime args (page counts, offsets, bank ids, fabric connection
    // trailers) derives from the tensor spec and socket topology, both of which are in the program
    // hash, so on a cache hit only these two base addresses can have moved. The socket config
    // address is outside the hash (see SendAsyncDeviceOperation::compute_program_hash).
    const uint32_t input_base_addr = input_buffer->address();
    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();

    for (const auto& sender_core_coord :
         collect_send_async_connections(mesh_socket, input_tensor, target_device).sender_core_coords) {
        tt::tt_metal::GetRuntimeArgs(
            program, send_async_reader_kernel_index, sender_core_coord)[send_async_reader_input_addr_arg_index] =
            input_base_addr;
        tt::tt_metal::GetRuntimeArgs(
            program,
            send_async_writer_kernel_index,
            sender_core_coord)[send_async_writer_socket_config_addr_arg_index] = socket_config_addr;
    }
}

}  // namespace ttnn::experimental::prim
