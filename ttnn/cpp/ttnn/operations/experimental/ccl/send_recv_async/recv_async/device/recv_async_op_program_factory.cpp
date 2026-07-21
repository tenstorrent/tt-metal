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
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

ProgramDescriptor build_receiver_program_descriptor(
    const RecvAsyncParams& operation_attributes,
    const Tensor& output_tensor,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    auto mesh_socket = operation_attributes.mesh_socket;
    auto* mesh_device = output_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : output_tensor.device();

    ProgramDescriptor desc;
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<uint32_t> connection_indices;

    for (uint32_t i = 0; i < socket_connection_config.size(); ++i) {
        const auto& connection = socket_connection_config[i];
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            receiver_core_coords.push_back(connection.receiver_core.core_coord);
            receiver_fabric_node_ids.push_back(
                output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord));
            sender_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord));
            connection_indices.push_back(i);
        }
    }
    uint32_t num_cores = receiver_core_coords.size();

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
        output_tensor.buffer()->alignment());
    auto output_page_size = output_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(output_page_size, max_alignment);
    auto total_num_pages = output_tensor.buffer()->num_pages();
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

    std::set<CoreRange> receiver_core_set;
    for (const auto& core : receiver_core_coords) {
        receiver_core_set.insert(CoreRange(core, core));
    }
    CoreRangeSet receiver_core_range_set(receiver_core_set);

    uint32_t packet_header_cb_num_pages = 1;  // One for sync
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;
    auto packet_header_cb_index = tt::CBIndex::c_0;

    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = receiver_core_range_set,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = packet_header_cb_index,
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size}},
    });

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    tt::CBIndex scratch_buffer_cb_index = tt::CBIndex::c_1;
    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

    if (socket_storage_in_dram) {
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
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = scratch_buffer_cb_index,
                .data_format = data_format,
                .page_size = socket_aligned_page_size}},
        });
    }

    if (!socket_storage_in_dram) {
        std::vector<uint32_t> writer_compile_args = {
            packet_header_cb_index,
            output_page_size,
            socket_block_size,
            socket_aligned_page_size,
            num_pages_per_packet,
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer_kernel_desc;
        writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/"
            "receiver_inplace_writer.cpp";
        writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_kernel_desc.core_ranges = receiver_core_range_set;
        writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
        writer_kernel_desc.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(writer_kernel_desc));
        const KernelHandle writer_kernel_id = desc.kernels.size() - 1;

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

            // Collect the fabric helper extras into a raw scratch vector; the
            // main writer args use an RTArgList so output_tensor.buffer() is
            // registered as a BufferBinding (patched on cache hit).
            std::vector<uint32_t> fabric_extras;
            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                fabric_extras);

            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(mesh_socket.get_config_buffer()->address());  // non-tensor; stable
            writer_rt_args.push_back(output_tensor.buffer());                      // Buffer* binding
            writer_rt_args.push_back(pages_for_this_core);
            writer_rt_args.push_back(page_start_offset);
            writer_rt_args.push_back(num_whole_packets);
            writer_rt_args.push_back(num_pages_remainder);
            writer_rt_args.append(fabric_extras);
            desc.kernels[writer_kernel_id].emplace_runtime_args(receiver_core_coord, writer_rt_args);
        }
    } else {
        std::vector<uint32_t> reader_compile_args = {
            packet_header_cb_index,
            scratch_buffer_cb_index,
            socket_block_size,
            socket_aligned_page_size,
            socket_storage_in_dram,
        };

        KernelDescriptor reader_kernel_desc;
        reader_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_reader.cpp";
        reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_kernel_desc.core_ranges = receiver_core_range_set;
        reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
        reader_kernel_desc.config = ReaderConfigDescriptor{};
        desc.kernels.push_back(std::move(reader_kernel_desc));
        const KernelHandle reader_kernel_id = desc.kernels.size() - 1;

        std::vector<uint32_t> writer_compile_args = {
            scratch_buffer_cb_index,
            output_page_size,
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        KernelDescriptor writer_kernel_desc;
        writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_writer.cpp";
        writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_kernel_desc.core_ranges = receiver_core_range_set;
        writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
        writer_kernel_desc.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(writer_kernel_desc));
        const KernelHandle writer_kernel_id = desc.kernels.size() - 1;

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

            uint32_t bank_id = 0;
            if (socket_storage_in_dram) {
                auto num_dram_banks = target_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
                bank_id = core_idx % num_dram_banks;
            } else {
                bank_id = target_device->allocator()->get_bank_ids_from_logical_core(
                    mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];
            }

            // Reader has only the non-tensor socket config buffer + fabric extras;
            // no tensor buffers, so plain raw uint32_t is fine.
            std::vector<uint32_t> reader_rt_args = {
                mesh_socket.get_config_buffer()->address(),
                bank_id,
                num_blocks,
                num_pages_per_block,
                block_remainder_pages,
            };

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                desc,
                receiver_core_coord,
                reader_rt_args);

            desc.kernels[reader_kernel_id].runtime_args.emplace_back(receiver_core_coord, std::move(reader_rt_args));

            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(output_tensor.buffer());  // Buffer* binding
            writer_rt_args.push_back(page_start_offset);
            writer_rt_args.push_back(pages_for_this_core);
            desc.kernels[writer_kernel_id].emplace_runtime_args(receiver_core_coord, writer_rt_args);
        }
    }

    return desc;
}

}  // namespace

WorkloadDescriptor RecvAsyncMeshWorkloadFactory::create_workload_descriptor(
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const ttnn::MeshCoordinateRangeSet receiver_coords =
        ttnn::send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
            tensor_coords, operation_attributes.mesh_socket);
    const auto receiver_coords_flat = receiver_coords.coords();

    WorkloadDescriptor wd;
    wd.programs.reserve(receiver_coords_flat.size());

    for (const auto& coord : receiver_coords_flat) {
        ProgramDescriptor desc = build_receiver_program_descriptor(operation_attributes, tensor_args, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
