// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

RecvAsyncMeshWorkloadFactory::cached_mesh_workload_t RecvAsyncMeshWorkloadFactory::create_mesh_workload(
    const RecvAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    ttnn::MeshCoordinateRangeSet workload_coords =
        ttnn::send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
            tensor_coords, operation_attributes.mesh_socket);
    for (const auto& coord : workload_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<RecvAsyncMeshWorkloadFactory::shared_variables_t>
RecvAsyncMeshWorkloadFactory::create_at(
    const RecvAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    auto mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    auto* mesh_device = output_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<uint32_t> connection_indices;

    // TODO #24995: Find appropriate receiver cores and fabric node IDs based on mesh socket configuration
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

    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
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

    auto receiver_core_range_set = CoreRangeSet(std::set<CoreRange>());
    for (const auto& core : receiver_core_coords) {
        receiver_core_range_set = receiver_core_range_set.merge(CoreRangeSet({CoreRange(core, core)}));
    }

    uint32_t packet_header_cb_num_pages = 1;  // One for sync
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, receiver_core_range_set, cb_packet_header_config);

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    tt::CBIndex scratch_buffer_cb_index = tt::CBIndex::c_1;
    bool socket_storage_in_dram =
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::DRAM;

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
        tt::tt_metal::CircularBufferConfig cb_scratch_buffer_config =
            tt::tt_metal::CircularBufferConfig(scratch_buffer_size, {{scratch_buffer_cb_index, data_format}})
                .set_page_size(scratch_buffer_cb_index, socket_aligned_page_size);
        CreateCircularBuffer(program, receiver_core_range_set, cb_scratch_buffer_config);
    }

    tt::tt_metal::KernelHandle reader_kernel = 0;
    tt::tt_metal::KernelHandle writer_kernel = 0;

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

        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/"
            "receiver_inplace_writer.cpp",
            receiver_core_range_set,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

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

            std::vector<uint32_t> writer_rt_args = {
                mesh_socket.get_config_buffer()->address(),  // socket_config_addr
                output_tensor.buffer()->address(),           // output_base_addr
                pages_for_this_core,                         // num_pages
                page_start_offset,                           // page_start_offset
                num_whole_packets,                           // num_whole_packets
                num_pages_remainder,                         // num_pages_remainder
            };

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
            tt::tt_fabric::append_fabric_connection_rt_args(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                program,
                receiver_core_coord,
                writer_rt_args);

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, receiver_core_coord, writer_rt_args);
        }
    } else {
        std::vector<uint32_t> reader_compile_args = {
            packet_header_cb_index,    // fabric_packet_header_cb_id
            scratch_buffer_cb_index,   // scratch_buffer_cb_id
            socket_block_size,         // socket_block_size
            socket_aligned_page_size,  // socket_page_size
            socket_storage_in_dram,    // socket_storage_in_dram
        };
        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_reader.cpp",
            receiver_core_range_set,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

        std::vector<uint32_t> writer_compile_args = {
            scratch_buffer_cb_index,  // scratch_buffer_cb_id
            output_page_size,         // page_size
        };
        writer_compile_args.insert(
            writer_compile_args.end(),
            output_accessor_compile_time_args.begin(),
            output_accessor_compile_time_args.end());

        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/receiver_writer.cpp",
            receiver_core_range_set,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

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

            std::vector<uint32_t> reader_rt_args = {
                mesh_socket.get_config_buffer()->address(),  // socket_config_addr
                bank_id,                                     // bank_id
                num_blocks,                                  // num_blocks
                num_pages_per_block,                         // num_pages_per_block
                block_remainder_pages,                       // block_remainder_pages
            };

            auto link_indices =
                tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
            TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

            uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];

            tt::tt_fabric::append_fabric_connection_rt_args(
                receiver_fabric_node_id,
                sender_fabric_node_id,
                selected_link_index,
                program,
                receiver_core_coord,
                reader_rt_args);

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, receiver_core_coord, reader_rt_args);

            std::vector<uint32_t> writer_rt_args = {
                output_tensor.buffer()->address(),  // output_base_addr
                page_start_offset,                  // start_page_index
                pages_for_this_core,                // num_pages
            };

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, receiver_core_coord, writer_rt_args);
        }
    }

    return {
        std::move(program),
        shared_variables_t{
            .receiver_core_coords = receiver_core_coords,
            .reader_kernel_id = reader_kernel,
            .writer_kernel_id = writer_kernel,
            .socket_storage_in_dram = socket_storage_in_dram,
        }};
}

void RecvAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& receiver_core_coords = shared_vars.receiver_core_coords;
        const auto& reader_kernel_id = shared_vars.reader_kernel_id;
        const auto& writer_kernel_id = shared_vars.writer_kernel_id;
        const auto& socket_storage_in_dram = shared_vars.socket_storage_in_dram;

        const auto& mesh_socket = operation_attributes.mesh_socket;
        const auto& output_tensor = tensor_args;

        if (!socket_storage_in_dram) {
            for (const auto& receiver_core_coord : receiver_core_coords) {
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, receiver_core_coord);

                writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
                writer_runtime_args[1] = output_tensor.buffer()->address();
            }
        } else {
            for (const auto& receiver_core_coord : receiver_core_coords) {
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, receiver_core_coord);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, receiver_core_coord);

                reader_runtime_args[0] = mesh_socket.get_config_buffer()->address();
                writer_runtime_args[0] = output_tensor.buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
