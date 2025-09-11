// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "point_to_point_device_op.hpp"

using tt::tt_fabric::get_fabric_node_id_from_physical_chip_id;

namespace ttnn::operations::point_to_point {

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore) {
    auto mesh_device = dynamic_cast<MeshDevice*>(output_tensors.at(0).device());

    const auto& send_coord = operation_attributes.send_coord;
    const auto& receive_coord = operation_attributes.receive_coord;
    const auto& intermediate_tensor = output_tensors.at(0);
    const auto& output_tensor = output_tensors.at(1);

    // basic accounting
    const uint32_t output_num_pages = data_movement::get_num_pages(output_tensor);
    const uint32_t output_page_size_bytes = output_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // figure out packets
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            output_tensor.dtype(), output_page_size_bytes, output_num_pages, l1_alignment);
    // distribute work
    const CoreCoord use_cores = {1, 1};

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    // program!
    tt::tt_metal::Program program{};

    tt::DataFormat inter_dataformat = tt::tt_metal::datatype_to_dataformat_converter(intermediate_tensor.dtype());

    // CB for packet headers
    constexpr auto packet_header_cb_id = tt::CBIndex::c_0;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_header_config);

    // Scratch CB for loading up pages that are collected into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, inter_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    // CB for sender reader->writer kernels
    constexpr auto receiver_cb_id = tt::CBIndex::c_2;
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;
    tt::tt_metal::CircularBufferConfig cb_receiver_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * output_page_size_bytes, {{receiver_cb_id, inter_dataformat}})
            .set_page_size(receiver_cb_id, output_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_receiver_config);

    const auto& topology = operation_attributes.topology;
    auto this_device = mesh_device->get_device(receive_coord);
    const auto this_fabric_id = get_fabric_node_id_from_physical_chip_id(this_device->id());
    const auto [num_hops, sender_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, receive_coord, send_coord, topology);

    std::vector<uint32_t> reader_ct_args = {packet_header_cb_id, packet_cb_id, receiver_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(reader_ct_args);
    tt::tt_metal::KernelHandle receive_unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // And the writer
    std::vector<uint32_t> writer_ct_args = {receiver_cb_id};
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    tt::tt_metal::KernelHandle receive_unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_unary_interleaved_start_id_gen.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    constexpr auto link_idx = 0;  // for single link implementation
    uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> receiver_cores;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_packets_per_core_group_1 * num_pages_per_packet;
        } else if (core_group_2.contains(c)) {
            increment = num_packets_per_core_group_2 * num_pages_per_packet;
        } else {
            continue;
        }
        increment = std::min(increment, output_num_pages - page_idx_start);
        page_idx_end += increment;

        std::vector<uint32_t> reader_runtime_args = {
            page_idx_start,
            page_idx_end,
            num_pages_per_packet,
            intermediate_tensor.mesh_buffer()->get_device_buffer(receive_coord)->address(),
            packet_size_bytes,
            output_page_size_bytes,
            num_page_segments,
            semaphore.address(),
            num_hops,
            sender_is_forward};

        if (sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, program, c, reader_runtime_args);
        }
        reader_runtime_args.emplace_back(!sender_is_forward);
        if (!sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, program, c, reader_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, receive_unary_reader_kernel_id, c, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args = {
            output_tensor.mesh_buffer()->get_device_buffer(receive_coord)->address(),
            increment,
            page_idx_start,
            output_page_size_bytes};

        tt::tt_metal::SetRuntimeArgs(program, receive_unary_writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
        receiver_cores.push_back(c);
    }

    return {
        std::move(program),
        PointToPointOp::SendReceive::shared_variables_t{
            .receive_unary_reader_kernel_id = receive_unary_reader_kernel_id,
            .receive_unary_writer_kernel_id = receive_unary_writer_kernel_id,
            .receiver_cores = receiver_cores,
            .semaphore = semaphore}};
}
}  // namespace ttnn::operations::point_to_point
