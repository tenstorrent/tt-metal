// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::get_fabric_node_id_from_physical_chip_id;

namespace ttnn::operations::point_to_point {

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore) {
    auto mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
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
    tt::tt_metal::Program program{};

    // CB for sender reader->writer kernels
    // Note this ID is hardcoded in the reader kernel
    constexpr auto sender_cb_id = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 2;
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * aligned_input_page_size_bytes, {{sender_cb_id, input_dataformat}})
            .set_page_size(sender_cb_id, aligned_input_page_size_bytes);
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
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_header_config);

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    // basic reader kernel set up
    std::vector<uint32_t> reader_ct_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::KernelHandle send_unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto this_device = mesh_device->get_device(send_coord);
    const auto this_fabric_id = get_fabric_node_id_from_physical_chip_id(this_device->id());

    const auto [num_hops, dst_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, send_coord, receive_coord, topology);

    std::vector<uint32_t> writer_ct_args = {sender_cb_id, packet_header_cb_id, packet_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(writer_ct_args);

    tt::tt_metal::KernelHandle send_unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    constexpr auto link_idx = 0;  // for single link implementation

    uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> sender_cores;
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

        const std::vector<uint32_t> reader_runtime_args = {
            input_tensor.mesh_buffer()->get_device_buffer(send_coord)->address(),
            increment,
            page_idx_start,
            input_page_size_bytes};
        tt::tt_metal::SetRuntimeArgs(program, send_unary_reader_kernel_id, c, reader_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {
            output_tensors.at(0).mesh_buffer()->get_device_buffer(receive_coord)->address(),
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
                this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
        }
        writer_runtime_args.emplace_back(!dst_is_forward);
        if (!dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, send_unary_writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
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
