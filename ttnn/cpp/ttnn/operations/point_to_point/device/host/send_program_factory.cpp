// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

namespace ttnn::operations::point_to_point {

tt::tt_metal::ProgramDescriptor send_program_factory(
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
    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // figure out packets
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(input_tensor.dtype(), input_page_size_bytes, input_num_pages, l1_alignment);

    // eventually add more cores for multi-link
    const CoreCoord use_cores = {1, 1};
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    // program!
    tt::tt_metal::ProgramDescriptor desc;

    // CB for sender reader->writer kernels
    // Note this ID is hardcoded in the reader kernel
    constexpr auto sender_cb_id = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 2;
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * aligned_input_page_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_cb_id),
            .data_format = input_dataformat,
            .page_size = aligned_input_page_size_bytes,
        }}},
    });

    // allocate space for packet headers for payload semaphore
    constexpr auto packet_header_cb_id = tt::CBIndex::c_1;
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

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = packet_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_cb_id),
            .data_format = input_dataformat,
            .page_size = packet_size_bytes,
        }}},
    });

    // basic reader kernel set up
    std::vector<uint32_t> reader_ct_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    const auto this_fabric_id = mesh_device->get_fabric_node_id(send_coord);

    const auto [num_hops, dst_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, send_coord, receive_coord, topology);

    std::vector<uint32_t> writer_ct_args = {sender_cb_id, packet_header_cb_id, packet_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // Push kernels onto desc.kernels so we can refer to them by stable index;
    // append_fabric_connection_rt_args() is templated on ProgramDescriptor and
    // indexes into desc.kernels via the KernelHandle.
    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    tt::tt_metal::KernelHandle send_unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle send_unary_writer_kernel_id = 1;

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
        increment = std::min(increment, input_num_pages - page_idx_start);
        page_idx_end += increment;

        // Reader RT args.  arg[0] is the input tensor's buffer address; push
        // it as Buffer* so the framework records a BufferBinding and patches
        // it on cache hit (no override_runtime_arguments).
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());
        reader_rt_args.push_back(increment);
        reader_rt_args.push_back(page_idx_start);
        reader_rt_args.push_back(input_page_size_bytes);
        desc.kernels[send_unary_reader_kernel_id].emplace_runtime_args(c, reader_rt_args);

        // Writer RT args.  Use a plain std::vector<uint32_t> first to interop
        // with append_fabric_connection_rt_args(), then promote to the
        // KernelDescriptor's RTArgList — index 0 (output buffer address) and
        // index 8 (semaphore address) become a Buffer* binding and an
        // absolute semaphore address, respectively.
        std::vector<uint32_t> writer_runtime_args = {
            output_tensors.at(0).buffer()->address(),  // placeholder, replaced via Buffer* below
            page_idx_start,
            page_idx_end,
            num_hops,
            input_page_size_bytes,
            packet_size_bytes,
            num_pages_per_packet,
            num_page_segments,
            semaphore.address(),
            dst_is_forward,
        };

        if (dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, desc, c, writer_runtime_args);
        }
        writer_runtime_args.emplace_back(!dst_is_forward);
        if (!dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, desc, c, writer_runtime_args);
        }

        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args_builder;
        writer_rt_args_builder.reserve(writer_runtime_args.size());
        writer_rt_args_builder.push_back(output_tensors.at(0).buffer());  // tensor_address0 as Buffer*
        for (size_t i = 1; i < writer_runtime_args.size(); ++i) {
            writer_rt_args_builder.push_back(writer_runtime_args[i]);
        }
        desc.kernels[send_unary_writer_kernel_id].emplace_runtime_args(c, writer_rt_args_builder);

        page_idx_start += increment;
    }

    return desc;
}
}  // namespace ttnn::operations::point_to_point
