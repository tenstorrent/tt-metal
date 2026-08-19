// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp"

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
    // Framing (packet dims, CB sizing, local packet-stride ct arg) must use the input tensor's OWN
    // buffer alignment (DRAM alignment on Blackhole is 64B, not L1's 16B) -- otherwise CBs sized for
    // an under-aligned page overflow once the reader/writer TensorAccessors below read/write the
    // buffer's true aligned page size.
    const uint32_t buffer_alignment = input_tensor.buffer()->alignment();

    // figure out packets
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        ::ttnn::ccl::dataflow::ccl_packet_dims(
            input_tensor.dtype(), input_page_size_bytes, input_num_pages, buffer_alignment);

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
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, buffer_alignment);
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

    // Packet headers are drawn from the fabric-L1 PacketHeaderPool by the kernel-side
    // FabricStreamSender (HeaderPolicy is gone — Pool is the idiomatic default), so no
    // packet-header CB is allocated here.

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_1;
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
        ::ttnn::ccl::dataflow::ccl_dm_route(mesh_device, send_coord, receive_coord, topology);

    std::vector<uint32_t> writer_ct_args = {sender_cb_id, packet_cb_id, buffer_alignment};
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
        // arg[3] is consumed by reader_unary_interleaved_start_id_gen.cpp's TensorAccessor ctor as an
        // override of TensorAccessorArgs::AlignedPageSize -- it must be the buffer's ALIGNED page size
        // (matches s.get_aligned_page_size() used for the noc_async_read below), not the raw/logical
        // page size, or odd-indexed pages address wrong when raw != aligned (e.g. unaligned RM DRAM).
        reader_rt_args.push_back(static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size()));
        desc.kernels[send_unary_reader_kernel_id].emplace_runtime_args(c, reader_rt_args);

        // Writer RT args. The fabric-connection block (built by the host helper, which owns its
        // wire layout) goes FIRST; the kernel consumes it with a cursor from 0 and reads the op's
        // own args after it — neither side hardcodes where the fabric block starts. Op args are
        // pushed as their natural types: Buffer* records a BufferBinding the framework patches on
        // a program-cache hit, so no placeholder/promotion pass is needed.
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        for (uint32_t arg : ::ttnn::ccl::dataflow::build_ccl_fabric_rt_args(
                 this_fabric_id, next_fabric_id, link_idx, desc, c, dst_is_forward)) {
            writer_rt_args.push_back(arg);
        }
        writer_rt_args.push_back(output_tensors.at(0).buffer());  // receiver base address (Buffer* binding)
        writer_rt_args.push_back(page_idx_start);
        writer_rt_args.push_back(page_idx_end);
        writer_rt_args.push_back(num_hops);
        writer_rt_args.push_back(input_page_size_bytes);
        writer_rt_args.push_back(packet_size_bytes);
        writer_rt_args.push_back(num_pages_per_packet);
        writer_rt_args.push_back(num_page_segments);
        writer_rt_args.push_back(semaphore.address());
        desc.kernels[send_unary_writer_kernel_id].emplace_runtime_args(c, writer_rt_args);

        page_idx_start += increment;
    }

    return desc;
}
}  // namespace ttnn::operations::point_to_point
