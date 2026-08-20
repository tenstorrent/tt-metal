// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_direct_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// The socket connections whose sender core sits on `target_device`, in socket-connection order.
// create_descriptor and override_runtime_arguments both walk this so the per-core runtime-arg
// ordering they assume stays identical.
struct SenderConnections {
    std::vector<CoreCoord> core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
};

SenderConnections collect_sender_connections(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket, const Tensor& input_tensor, IDevice* target_device) {
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    SenderConnections connections;
    for (const auto& connection : mesh_socket.get_config().socket_connection_config) {
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            connections.core_coords.push_back(connection.sender_core.core_coord);
            connections.sender_fabric_node_ids.push_back(
                input_tensor.device()->get_fabric_node_id(connection.sender_core.device_coord));
            connections.receiver_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord));
        }
    }
    return connections;
}

// Descriptor kernel indices, fixed by the push order at the end of create_descriptor.
constexpr uint32_t reader_kernel_index = 0;
constexpr uint32_t writer_kernel_index = 1;

}  // namespace

ProgramDescriptor SendDirectAsyncProgramFactory::create_descriptor(
    const SendDirectAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(input_tensor, mesh_dispatch_coordinate, "send_direct_async");

    auto connections = collect_sender_connections(mesh_socket, input_tensor, target_device);
    const auto& sender_core_coords = connections.core_coords;
    const auto& sender_fabric_node_ids = connections.sender_fabric_node_ids;
    const auto& receiver_fabric_node_ids = connections.receiver_fabric_node_ids;

    uint32_t num_cores = sender_core_coords.size();
    // This device holds no sender core of the socket, so it has no work. An empty descriptor tells
    // the framework to skip this coordinate.
    if (num_cores == 0) {
        return ProgramDescriptor{};
    }

    // cores must not exceed available fabric links
    {
        auto available_link_indices =
            tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_ids[0], sender_fabric_node_ids[0]);
        uint32_t num_available_links = available_link_indices.size();

        TT_FATAL(
            num_cores <= num_available_links,
            "Cannot create {} receiver-sender pairs with only {} available fabric links between devices. "
            "Reduce the number of cores per device.",
            num_cores,
            num_available_links);
    }

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());

    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);
    auto total_num_pages = input_tensor.buffer()->num_pages();

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    auto* input_buffer = input_tensor.buffer();
    const bool is_interleaved = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
    uint32_t num_banks = input_buffer->allocator()->get_num_banks(input_buffer->buffer_type());

    auto fabric_max_payload_size = tt::round_down(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    // Bank-contiguous packing: in an interleaved tensor, pages whose indices differ by num_banks sit
    // in the same bank at consecutive slots, so they are contiguous in memory and can be gathered by
    // one read and forwarded in one fabric packet. Restricted to DRAM so the super-block CB stays
    // small (~12 banks); L1-interleaved has one bank per core and would need an impractical CB.
    const bool is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    uint32_t enable_bank_packing = (is_interleaved && is_dram && num_pages_per_packet > 1 && num_banks > 1) ? 1u : 0u;

    // Small pages (num_pages_per_packet > 0): pack multiple whole pages into one fabric packet.
    // Large pages (num_pages_per_packet == 0): split a single page across multiple fabric packets.
    uint32_t num_whole_packets_per_page = 0, partial_packet_size = 0;
    if (num_pages_per_packet == 0) {
        num_whole_packets_per_page = input_page_size / fabric_max_payload_size;
        partial_packet_size = input_page_size % fabric_max_payload_size;
    }

    uint32_t handshake_page_size = ttnn::send_recv_utils::handshake_page_size(max_alignment);

    uint32_t cb_num_pages = 2;
    // Under bank packing each CB entry holds one super-block: num_banks regions of
    // num_pages_per_packet pages, at input_page_size stride and contiguous within a bank.
    uint32_t cb_page_size = enable_bank_packing
                                ? static_cast<uint32_t>(num_banks * num_pages_per_packet * input_page_size)
                                : fabric_max_payload_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    std::set<CoreRange> sender_core_ranges;
    for (const auto& core : sender_core_coords) {
        sender_core_ranges.insert(CoreRange(core));
    }
    CoreRangeSet sender_core_range_set(sender_core_ranges);

    ProgramDescriptor desc;

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
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

    // Page 0 is the dest-info landing zone the receiver writes back into, page 1 stages the
    // advertise payload pushed to the receiver over the socket.
    constexpr uint8_t handshake_cb_index = tt::CBIndex::c_2;
    uint32_t handshake_cb_num_pages = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = handshake_cb_num_pages * handshake_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = handshake_cb_index,
            .data_format = tt::DataFormat::UInt32,
            .page_size = handshake_page_size,
        }}},
    });

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

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/device/kernels/"
        "sender_direct_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = sender_core_range_set;
    reader.compile_time_args = std::move(reader_compile_args);
    reader.config = ReaderConfigDescriptor{};

    // The writer addresses the receiver's output tensor, whose layout matches the input tensor.
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,               // cb0_id
        packet_header_cb_index,      // fabric_packet_header_cb_id
        handshake_cb_index,          // handshake_cb_id
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

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/device/kernels/"
        "sender_direct_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = sender_core_range_set;
    writer.compile_time_args = std::move(writer_compile_args);
    writer.config = WriterConfigDescriptor{};

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& sender_core_coord = sender_core_coords[core_idx];
        uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

        uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
        uint32_t num_whole_packets = 0, num_pages_remainder = 0;
        if (num_pages_per_packet > 0) {
            num_whole_packets = pages_for_this_core / num_pages_per_packet;
            num_pages_remainder = pages_for_this_core % num_pages_per_packet;
        }

        // Both addresses below are re-applied by override_runtime_arguments instead of being declared
        // as Buffer* bindings: the socket config buffer is not tensor-backed, so the binding fast
        // path would patch the input address and leave the socket address frozen at first miss.
        reader.runtime_args.emplace_back(
            sender_core_coord,
            KernelDescriptor::CoreRuntimeArgs{
                input_tensor.buffer()->address(),  // smuggled-rta-ok: patched in override_runtime_arguments
                pages_for_this_core,               // num_pages
                page_start_offset,                 // page_start_offset
                num_whole_packets,                 // num_whole_packets
                num_pages_remainder,               // num_pages_remainder
            });

        std::vector<uint32_t> writer_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            pages_for_this_core,                         // num_pages
            page_start_offset,                           // page_start_offset
            num_whole_packets,                           // num_whole_packets
            num_pages_remainder,                         // num_pages_remainder
        };

        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];
        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            selected_link_index,
            desc,
            sender_core_coord,
            writer_rt_args);

        writer.runtime_args.emplace_back(sender_core_coord, std::move(writer_rt_args));
    }

    // Kernel order fixes the descriptor kernel indices: 0 = reader, 1 = writer.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));

    return desc;
}

void SendDirectAsyncProgramFactory::override_runtime_arguments(
    Program& program,
    const SendDirectAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& input_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(input_tensor, mesh_dispatch_coordinate, "send_direct_async");

    // Everything else in the runtime args (page counts, offsets, fabric connection trailers) derives
    // from the tensor spec and socket topology, both of which are in the program hash — so on a cache
    // hit only these two base addresses can have moved.
    const uint32_t input_base_addr = input_tensor.buffer()->address();
    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();

    for (const auto& sender_core_coord :
         collect_sender_connections(mesh_socket, input_tensor, target_device).core_coords) {
        GetRuntimeArgs(program, reader_kernel_index, sender_core_coord)[0] = input_base_addr;
        GetRuntimeArgs(program, writer_kernel_index, sender_core_coord)[0] = socket_config_addr;
    }
}

}  // namespace ttnn::experimental::prim
