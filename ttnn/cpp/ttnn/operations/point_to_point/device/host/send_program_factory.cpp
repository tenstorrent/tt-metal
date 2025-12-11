// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::point_to_point {

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
    const auto& topology = operation_attributes.topology;
    const auto& input_tensor = tensor_args.input_tensor;

    // basic accounting
    const uint32_t input_num_pages = data_movement::get_num_pages(input_tensor);
    printf("Input number of pages: %u\n", input_num_pages);
    const uint32_t input_page_size_bytes = input_tensor.buffer()->page_size();
    printf("Input tensor page size (bytes): %u\n", input_page_size_bytes);
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    uint32_t num_links = 2;

    // figure out packets - pass num_links to compute per-link packet sizing
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(input_tensor.dtype(), input_page_size_bytes, input_num_pages, l1_alignment);
    printf("packet size bytes: %u\n", packet_size_bytes);
    uint32_t num_workers_per_link = 1;
    const auto [all_cores, cores] = ttnn::ccl::choose_worker_cores(
        num_links, num_workers_per_link, mesh_device, std::nullopt, CoreCoord{0, 0}, std::nullopt);

    printf("all core are:\n");
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        printf("  core (%zu,%zu)\n", c.x, c.y);
    }
    // program!
    tt::tt_metal::Program program{};
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    printf("Input tensor tile shape: (%u, %u)\n", tile_height, tile_width);
    const auto tiny_tile = tt::tt_metal::Tile({tile_height, tile_width});

    // CB for sender reader->writer kernels
    // Note this ID is hardcoded in the reader kernel
    constexpr auto sender_cb_id = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 2;
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    printf("Aligned input page size (bytes): %u\n", aligned_input_page_size_bytes);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * aligned_input_page_size_bytes, {{sender_cb_id, input_dataformat}})
            .set_page_size(sender_cb_id, aligned_input_page_size_bytes)
            .set_tile_dims(sender_cb_id, tiny_tile);
    CreateCircularBuffer(program, all_cores, cb_sender_config);

    // allocate space for packet headers for payload sempahore
    constexpr auto packet_header_cb_id = tt::CBIndex::c_1;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id, tiny_tile);
    CreateCircularBuffer(program, all_cores, cb_header_config);

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes)
            .set_tile_dims(packet_cb_id, tiny_tile);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    // basic reader kernel set up
    std::vector<uint32_t> reader_ct_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::KernelHandle send_unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    const auto this_fabric_id = mesh_device->get_fabric_node_id(send_coord);

    const auto [num_hops, dst_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, send_coord, receive_coord, topology);

    std::vector<uint32_t> writer_ct_args = {sender_cb_id, packet_header_cb_id, packet_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelHandle send_unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    uint32_t link_idx = 0;  // for single link implementation
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    std::vector<CoreCoord> sender_cores;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t page_idx_start = (link_idx * base_pages_per_worker) + std::min(link_idx, remainder);
        uint32_t page_idx_end = ((link_idx + 1) * base_pages_per_worker) + std::min(link_idx + 1, remainder);

        // Calculate packet index offset for this link's portion of the intermediate buffer
        uint32_t packet_idx_offset = link_idx > 0 ? (page_idx_start / num_pages_per_packet) : 0;

        printf(
            "for core (%zu,%zu), page_idx_start: %u, page_idx_end: %u, packet_idx_offset: %u\n",
            c.x,
            c.y,
            page_idx_start,
            page_idx_end,
            packet_idx_offset);

        uint32_t increment = page_idx_end - page_idx_start;
        const std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(), increment, page_idx_start, input_page_size_bytes};
        tt::tt_metal::SetRuntimeArgs(program, send_unary_reader_kernel_id, c, reader_runtime_args);

        // auto core_xy = mesh_device->worker_core_from_logical_core(c);
        std::vector<uint32_t> writer_runtime_args = {
            output_tensors.at(0).buffer()->address(),
            page_idx_start,
            page_idx_end,
            num_hops,
            input_page_size_bytes,
            packet_size_bytes,
            num_pages_per_packet,
            num_page_segments,
            semaphore.address(),
            // core_xy.x,
            // core_xy.y,
            dst_is_forward,
            packet_idx_offset,  // Add packet index offset for this link
        };

        if (dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
        }
        writer_runtime_args.emplace_back(!dst_is_forward);
        if (!dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, send_unary_writer_kernel_id, c, writer_runtime_args);

        link_idx++;
        sender_cores.push_back(c);
    }

    return {
        std::move(program),
        PointToPointOp::SendReceive::shared_variables_t{
            .send_unary_reader_kernel_id = send_unary_reader_kernel_id,
            .send_unary_writer_kernel_id = send_unary_writer_kernel_id,
            .sender_cores = sender_cores,
            .semaphore = semaphore}};
}
}  // namespace ttnn::operations::point_to_point
