// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// The socket connections whose receiver core sits on `target_device`, in socket-connection order.
// create_descriptor and override_runtime_arguments both walk this so the per-core runtime-arg
// ordering they assume stays identical.
struct RecvAsyncConnections {
    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
};

RecvAsyncConnections collect_recv_async_connections(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const Tensor& output_tensor,
    tt::tt_metal::IDevice* target_device) {
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    RecvAsyncConnections connections;
    connections.receiver_core_coords.reserve(socket_connection_config.size());
    connections.sender_fabric_node_ids.reserve(socket_connection_config.size());
    connections.receiver_fabric_node_ids.reserve(socket_connection_config.size());

    // TODO #24995: Find appropriate receiver cores and fabric node IDs based on mesh socket configuration
    for (const auto& connection : socket_connection_config) {
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            connections.receiver_core_coords.push_back(connection.receiver_core.core_coord);
            connections.receiver_fabric_node_ids.push_back(
                output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord));
            connections.sender_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord));
        }
    }
    return connections;
}

bool recv_async_socket_storage_in_dram(const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;
}

// Descriptor kernel indices and re-applied runtime-arg slots, fixed by the push order in
// create_descriptor. The kernel layout depends on the socket storage type, which is in the hash.
//
// L1 socket storage: a single in-place writer reads the socket FIFO and writes the output tensor.
constexpr uint32_t recv_async_l1_writer_kernel_index = 0;
constexpr uint32_t recv_async_l1_writer_socket_config_addr_arg_index = 0;
constexpr uint32_t recv_async_l1_writer_output_addr_arg_index = 1;
// DRAM socket storage: a reader drains the socket into a scratch CB, a writer stores it to the output.
constexpr uint32_t recv_async_dram_reader_kernel_index = 0;
constexpr uint32_t recv_async_dram_writer_kernel_index = 1;
constexpr uint32_t recv_async_dram_reader_socket_config_addr_arg_index = 0;
constexpr uint32_t recv_async_dram_writer_output_addr_arg_index = 0;

}  // namespace

tt::tt_metal::ProgramDescriptor RecvAsyncProgramFactory::create_descriptor(
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;

    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(output_tensor, mesh_dispatch_coordinate, "recv_async");

    auto connections = collect_recv_async_connections(mesh_socket, output_tensor, target_device);
    const auto& receiver_core_coords = connections.receiver_core_coords;
    const auto& sender_fabric_node_ids = connections.sender_fabric_node_ids;
    const auto& receiver_fabric_node_ids = connections.receiver_fabric_node_ids;

    uint32_t num_cores = receiver_core_coords.size();
    // This device holds no receiver core of the socket, so it has no work. An empty descriptor tells
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

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "recv_async: output tensor buffer is null");

    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        output_buffer->alignment());
    auto output_page_size = output_buffer->aligned_page_size();
    auto socket_aligned_page_size = tt::align(output_page_size, max_alignment);
    auto total_num_pages = output_buffer->num_pages();
    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        max_alignment);
    auto num_pages_per_packet = fabric_max_payload_size / socket_aligned_page_size;

    uint32_t pages_per_core = total_num_pages / num_cores;
    uint32_t remainder_pages = total_num_pages % num_cores;

    uint32_t socket_block_size = 0;
    if (num_pages_per_packet > 0) {
        socket_block_size = num_pages_per_packet * socket_aligned_page_size;
    } else {
        socket_block_size = socket_aligned_page_size;
    }

    auto receiver_core_range_set = CoreRangeSet(std::set<CoreRange>());
    for (const auto& core : receiver_core_coords) {
        receiver_core_range_set = receiver_core_range_set.merge(CoreRangeSet({CoreRange(core, core)}));
    }

    ProgramDescriptor desc;

    uint32_t packet_header_cb_num_pages = 1;  // One for sync
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;

    constexpr uint8_t packet_header_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = receiver_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = packet_header_cb_index,
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size,
        }}},
    });

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_buffer);
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    constexpr uint8_t scratch_buffer_cb_index = tt::CBIndex::c_1;
    bool socket_storage_in_dram = recv_async_socket_storage_in_dram(mesh_socket);

    if (socket_storage_in_dram) {
        // For DRAM mode, scratch buffer size should be based on packet size, not total pages per core
        // This matches the original single-core logic: 2 * num_pages_per_block * socket_aligned_page_size
        uint32_t num_pages_per_block = 0;
        if (num_pages_per_packet > 0) {
            num_pages_per_block = num_pages_per_packet;
        } else {
            num_pages_per_block = 1;
        }
        uint32_t scratch_buffer_size = 2 * num_pages_per_block * socket_aligned_page_size;

        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_buffer_size,
            .core_ranges = receiver_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = scratch_buffer_cb_index,
                .data_format = data_format,
                .page_size = socket_aligned_page_size,
            }}},
        });
    }

    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();

    if (!socket_storage_in_dram) {
        std::vector<uint32_t> writer_compile_args = {
            packet_header_cb_index,    // fabric_packet_header_cb_id
            output_page_size,          // output_page_size
            socket_block_size,         // socket_block_size
            socket_aligned_page_size,  // socket_page_size
            num_pages_per_packet,      // num_pages_per_packet
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer;
        writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/"
            "receiver_inplace_writer.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = receiver_core_range_set;
        writer.compile_time_args = std::move(writer_compile_args);
        writer.named_compile_time_args = {{"fabric_packet_header_cb_id", packet_header_cb_index}};
        writer.config = WriterConfigDescriptor{};

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            const auto& receiver_core_coord = receiver_core_coords[core_idx];
            const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
            const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

            uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

            uint32_t page_start_offset = 0;
            for (uint32_t prev_idx = 0; prev_idx < core_idx; ++prev_idx) {
                uint32_t prev_pages = pages_per_core + (prev_idx < remainder_pages ? 1 : 0);
                page_start_offset += prev_pages;
            }

            uint32_t num_whole_packets = 0, num_pages_remainder = 0;
            if (num_pages_per_packet > 0) {
                num_whole_packets = pages_for_this_core / num_pages_per_packet;
                num_pages_remainder = pages_for_this_core % num_pages_per_packet;
            }

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
            std::vector<uint32_t> fabric_connection_rt_args;
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                fabric_connection_rt_args);

            // The socket config buffer is not tensor-backed and its address is not in the program hash,
            // so it cannot be a Buffer* binding; override_runtime_arguments re-applies it on every hit.
            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(socket_config_addr);  // smuggled-rta-ok: re-applied via override_runtime_arguments
            writer_rt_args.push_back(output_buffer);       // output_base_addr
            writer_rt_args.push_back(pages_for_this_core);  // num_pages
            writer_rt_args.push_back(page_start_offset);    // page_start_offset
            writer_rt_args.push_back(num_whole_packets);    // num_whole_packets
            writer_rt_args.push_back(num_pages_remainder);  // num_pages_remainder
            writer_rt_args.append(fabric_connection_rt_args);
            writer.emplace_runtime_args(receiver_core_coord, writer_rt_args);
        }

        TT_FATAL(
            desc.kernels.size() == recv_async_l1_writer_kernel_index, "recv_async: L1 writer kernel index mismatch");
        desc.kernels.push_back(std::move(writer));
    } else {
        std::vector<uint32_t> reader_compile_args = {
            packet_header_cb_index,    // fabric_packet_header_cb_id
            scratch_buffer_cb_index,   // scratch_buffer_cb_id
            socket_block_size,         // socket_block_size
            socket_aligned_page_size,  // socket_page_size
            socket_storage_in_dram,    // socket_storage_in_dram
        };

        KernelDescriptor reader;
        reader.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = receiver_core_range_set;
        reader.compile_time_args = std::move(reader_compile_args);
        reader.named_compile_time_args = {
            {"fabric_packet_header_cb_id", packet_header_cb_index},
            {"scratch_buffer_cb_id", scratch_buffer_cb_index},
        };
        reader.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_compile_args = {
            scratch_buffer_cb_index,  // scratch_buffer_cb_id
            output_page_size,         // page_size
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer;
        writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_writer.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = receiver_core_range_set;
        writer.compile_time_args = std::move(writer_compile_args);
        writer.named_compile_time_args = {{"scratch_buffer_cb_id", scratch_buffer_cb_index}};
        writer.config = WriterConfigDescriptor{};

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            const auto& receiver_core_coord = receiver_core_coords[core_idx];
            const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
            const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

            uint32_t pages_for_this_core = pages_per_core + (core_idx < remainder_pages ? 1 : 0);

            uint32_t page_start_offset = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);

            uint32_t num_whole_packets = 0, num_pages_remainder_core = 0;
            if (num_pages_per_packet > 0) {
                num_whole_packets = pages_for_this_core / num_pages_per_packet;
                num_pages_remainder_core = pages_for_this_core % num_pages_per_packet;
            }

            uint32_t num_blocks = 0, num_pages_per_block = 0, block_remainder_pages = 0;
            if (num_pages_per_packet > 0) {
                num_blocks = num_whole_packets;
                num_pages_per_block = num_pages_per_packet;
                block_remainder_pages = num_pages_remainder_core;
            } else {
                num_blocks = pages_for_this_core;
                num_pages_per_block = 1;
                block_remainder_pages = 0;
            }

            // TODO #24995: This should be derived from the expected tensor/socket configuration
            uint32_t bank_id = 0;
            if (socket_storage_in_dram) {
                // Assign DRAM banks in round-robin for each receiver core
                auto num_dram_banks = target_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
                bank_id = core_idx % num_dram_banks;
            } else {
                // L1 mode: use logical core mapping
                bank_id = target_device->allocator()->get_bank_ids_from_logical_core(
                    mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
            }

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];

            std::vector<uint32_t> fabric_connection_rt_args;
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                fabric_connection_rt_args);

            // The socket config buffer is not tensor-backed and its address is not in the program hash,
            // so it cannot be a Buffer* binding; override_runtime_arguments re-applies it on every hit.
            KernelDescriptor::RTArgList reader_rt_args;
            reader_rt_args.push_back(socket_config_addr);  // smuggled-rta-ok: re-applied via override_runtime_arguments
            reader_rt_args.push_back(bank_id);             // bank_id
            reader_rt_args.push_back(num_blocks);          // num_blocks
            reader_rt_args.push_back(num_pages_per_block);    // num_pages_per_block
            reader_rt_args.push_back(block_remainder_pages);  // block_remainder_pages
            reader_rt_args.append(fabric_connection_rt_args);
            reader.emplace_runtime_args(receiver_core_coord, reader_rt_args);

            writer.emplace_runtime_args(
                receiver_core_coord,
                {
                    output_buffer,        // output_base_addr
                    page_start_offset,    // start_page_index
                    pages_for_this_core,  // num_pages
                });
        }

        TT_FATAL(
            desc.kernels.size() == recv_async_dram_reader_kernel_index,
            "recv_async: DRAM reader kernel index mismatch");
        desc.kernels.push_back(std::move(reader));
        TT_FATAL(
            desc.kernels.size() == recv_async_dram_writer_kernel_index,
            "recv_async: DRAM writer kernel index mismatch");
        desc.kernels.push_back(std::move(writer));
    }

    return desc;
}

void RecvAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    tt::tt_metal::IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(output_tensor, mesh_dispatch_coordinate, "recv_async");

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "recv_async: output tensor buffer is null");

    // Everything else in the runtime args (page counts, offsets, bank ids, fabric connection
    // trailers) derives from the tensor spec and socket topology, both of which are in the program
    // hash, so on a cache hit only these two base addresses can have moved. The socket config
    // address is outside the hash (see RecvAsyncDeviceOperation::compute_program_hash). The output
    // tensor is also the op's return value; it is patched here by role, so the alias is harmless.
    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();
    const uint32_t output_base_addr = output_buffer->address();
    const auto receiver_core_coords =
        collect_recv_async_connections(mesh_socket, output_tensor, target_device).receiver_core_coords;

    if (!recv_async_socket_storage_in_dram(mesh_socket)) {
        for (const auto& receiver_core_coord : receiver_core_coords) {
            auto& writer_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, recv_async_l1_writer_kernel_index, receiver_core_coord);
            writer_runtime_args[recv_async_l1_writer_socket_config_addr_arg_index] = socket_config_addr;
            writer_runtime_args[recv_async_l1_writer_output_addr_arg_index] = output_base_addr;
        }
    } else {
        for (const auto& receiver_core_coord : receiver_core_coords) {
            tt::tt_metal::GetRuntimeArgs(
                program,
                recv_async_dram_reader_kernel_index,
                receiver_core_coord)[recv_async_dram_reader_socket_config_addr_arg_index] = socket_config_addr;
            tt::tt_metal::GetRuntimeArgs(
                program,
                recv_async_dram_writer_kernel_index,
                receiver_core_coord)[recv_async_dram_writer_output_addr_arg_index] = output_base_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
