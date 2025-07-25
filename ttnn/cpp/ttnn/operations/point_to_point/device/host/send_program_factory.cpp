// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/fabric.hpp>
#include <tt-metalium/work_split.hpp>
#include "point_to_point_device_op.hpp"

using ttnn::operations::ccl::common::get_linearized_index;

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::get_fabric_node_id_from_physical_chip_id;

namespace ttnn::operations::point_to_point {
namespace detail {

auto one_d_fabric_routing_vector(const MeshCoordinate& src_coord, const MeshCoordinate& dest_coord) {
    // transmit along row
    if (src_coord[0] == dest_coord[0]) {
        constexpr auto dim = 1;
        const int hops = dest_coord[dim] - src_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    // transmit along col
    else if (src_coord[1] == dest_coord[1]) {
        constexpr auto dim = 0;
        const int hops = dest_coord[dim] - src_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    } else {
        TT_THROW("Routing coordinates {} and {} invalid for 1D fabric", src_coord, dest_coord);
        return std::make_tuple(0, false, 0);
    }
}

auto one_d_fabric_routing(
    const MeshDevice* mesh_device,
    const MeshCoordinate& src_coord,
    const MeshCoordinate& dest_coord,
    const ::ttnn::ccl::Topology& topology) {
    const auto& mesh_shape = mesh_device->get_view().shape();

    // sign indicates direction, however fabrics' forward/backward concept is reversed
    const auto [line_hops, line_is_forward, dim] = one_d_fabric_routing_vector(src_coord, dest_coord);

    TT_FATAL(line_hops != 0, "Should not be send/receiving to the same device");

    auto get_neighbor_id = [&src_coord, &mesh_device, &mesh_shape, dim](
                               bool is_forward, MeshCoordinate::BoundaryMode boundary_mode) {
        const auto neighbor_coord = src_coord.get_neighbor(mesh_shape, (is_forward ? 1 : -1), dim, boundary_mode);

        TT_FATAL(neighbor_coord.has_value(), "Can't find neighbor for {}", src_coord);
        auto next_device = mesh_device->get_device(neighbor_coord.value());
        const auto next_fabric_id = get_fabric_node_id_from_physical_chip_id(next_device->id());

        TT_FATAL(next_device != nullptr, "Did not find next device");
        return next_fabric_id;
    };

    if (topology == ::ttnn::ccl::Topology::Ring) {
        int ring_hops = line_hops + (line_hops < 0 ? -1 : 1) * mesh_shape[dim];

        if (std::abs(ring_hops) < std::abs(line_hops)) {
            bool ring_is_forward = (ring_hops > 0);

            const auto next_fabric_id = get_neighbor_id(ring_is_forward, MeshCoordinate::BoundaryMode::WRAP);
            return std::make_tuple(std::abs(ring_hops), !ring_is_forward, next_fabric_id);
        }
    }
    const auto next_fabric_id = get_neighbor_id(line_is_forward, MeshCoordinate::BoundaryMode::NONE);
    return std::make_tuple(line_hops, !line_is_forward, next_fabric_id);
}
}  // namespace detail

ttnn::device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensors) {
    auto mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());
    const auto& topology = operation_attributes.topology;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& receiver_semaphore = operation_attributes.receiver_semaphore;

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
    tt::tt_metal::CBHandle cb_sender_handle = CreateCircularBuffer(program, all_cores, cb_sender_config);

    // allocate space for packet headers for payload sempahore
    constexpr auto packet_header_cb_id = tt::CBIndex::c_1;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);
    auto cb_header_handle = CreateCircularBuffer(program, all_cores, cb_header_config);

    // Scratch CB for coalescing pages into packets
    constexpr auto packet_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    tt::tt_metal::CBHandle cb_cb_handle = CreateCircularBuffer(program, all_cores, cb_packet_config);

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // basic reader kernel set up
    tt::tt_metal::KernelHandle send_unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({input_is_dram}));

    auto this_device = mesh_device->get_device(send_coord);
    const auto this_fabric_id = get_fabric_node_id_from_physical_chip_id(this_device->id());

    const auto [num_hops, dst_is_forward, next_fabric_id] =
        detail::one_d_fabric_routing(mesh_device, send_coord, receive_coord, topology);

    const bool output_is_dram = output_tensors.at(0).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const std::vector<uint32_t> writer_ct_args = {
        sender_cb_id, packet_cb_id, packet_header_cb_id, output_is_dram, l1_alignment};

    tt::tt_metal::KernelHandle send_unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    constexpr auto link_idx = 0;  // equivalent to num_links = 0

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
            receiver_semaphore.address(),
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

    // !TODO implement program cache #23425
    return {
        std::move(program),
        PointToPointOp::SendReceive::shared_variables_t{
            .send_unary_reader_kernel_id = send_unary_reader_kernel_id,
            .send_unary_writer_kernel_id = send_unary_writer_kernel_id,
            .sender_cores = sender_cores}};
}
}  // namespace ttnn::operations::point_to_point
