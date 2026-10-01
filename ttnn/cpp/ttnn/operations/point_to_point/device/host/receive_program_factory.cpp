// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "point_to_point_device_op.hpp"

namespace ttnn::operations::point_to_point {

tt::tt_metal::ProgramDescriptor receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(output_tensors.at(0).device());

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
    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat inter_dataformat = tt::tt_metal::datatype_to_dataformat_converter(intermediate_tensor.dtype());

    // CB for packet headers
    constexpr auto packet_header_cb_id = tt::CBIndex::c_0;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb_id),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // Scratch CB for loading up pages that are collected into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_1;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = packet_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_cb_id),
            .data_format = inter_dataformat,
            .page_size = packet_size_bytes,
        }}},
    });

    // CB for sender reader->writer kernels
    constexpr auto receiver_cb_id = tt::CBIndex::c_2;
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * output_page_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(receiver_cb_id),
            .data_format = inter_dataformat,
            .page_size = output_page_size_bytes,
        }}},
    });

    const auto& topology = operation_attributes.topology;
    const auto this_fabric_id = mesh_device->get_fabric_node_id(receive_coord);
    const auto [num_hops, sender_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, receive_coord, send_coord, topology);

    std::vector<uint32_t> reader_ct_args = {packet_header_cb_id, packet_cb_id, receiver_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // And the writer
    std::vector<uint32_t> writer_ct_args = {receiver_cb_id};
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_unary_interleaved_start_id_gen.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // Push kernels into desc.kernels first so we can refer to them by stable
    // index — append_fabric_connection_rt_args() indexes desc.kernels via the
    // KernelHandle for its ProgramDescriptor overload.
    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    tt::tt_metal::KernelHandle receive_unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle receive_unary_writer_kernel_id = 1;

    constexpr auto link_idx = 0;  // for single link implementation
    uint32_t page_idx_start = 0, page_idx_end = 0;
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

        // Build reader RT args via plain vector to interop with the fabric
        // helper, then promote to the KernelDescriptor's RTArgList — index 3
        // (intermediate buffer address) becomes a Buffer* binding, and index
        // 7 stays as the GlobalSemaphore's absolute address.
        std::vector<uint32_t> reader_runtime_args = {
            page_idx_start,
            page_idx_end,
            num_pages_per_packet,
            intermediate_tensor.buffer()->address(),  // placeholder, replaced via Buffer* below
            packet_size_bytes,
            output_page_size_bytes,
            num_page_segments,
            semaphore.address(),
            num_hops,
            sender_is_forward};

        if (sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, desc, c, reader_runtime_args);
        }
        reader_runtime_args.emplace_back(!sender_is_forward);
        if (!sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, desc, c, reader_runtime_args);
        }

        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args_builder;
        reader_rt_args_builder.reserve(reader_runtime_args.size());
        for (size_t i = 0; i < reader_runtime_args.size(); ++i) {
            if (i == 3) {
                reader_rt_args_builder.push_back(intermediate_tensor.buffer());
            } else {
                reader_rt_args_builder.push_back(reader_runtime_args[i]);
            }
        }
        desc.kernels[receive_unary_reader_kernel_id].emplace_runtime_args(c, reader_rt_args_builder);

        // Writer RT args.  Index 0 is the output buffer's base address —
        // push as Buffer* so the framework records a BufferBinding.
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(output_tensor.buffer());
        writer_rt_args.push_back(increment);
        writer_rt_args.push_back(page_idx_start);
        writer_rt_args.push_back(output_page_size_bytes);
        desc.kernels[receive_unary_writer_kernel_id].emplace_runtime_args(c, writer_rt_args);

        page_idx_start += increment;
    }

    return desc;
}
}  // namespace ttnn::operations::point_to_point
