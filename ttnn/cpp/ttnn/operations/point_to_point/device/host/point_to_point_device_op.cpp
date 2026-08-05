// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"

#include "point_to_point_device_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::point_to_point {

namespace detail {

AlignedPacketDims compute_aligned_packet_dims(
    const DataType& dtype, const uint32_t page_size_bytes, const uint32_t num_pages, const uint32_t alignment) {
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const uint32_t max_packet_size_bytes =
        dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const uint32_t aligned_page_size_bytes = tt::round_up(page_size_bytes, alignment);

    uint32_t num_page_segments, max_num_pages_per_packet, packet_size_bytes, total_packets;
    if (aligned_page_size_bytes <= max_packet_size_bytes) {
        num_page_segments = 1;
        max_num_pages_per_packet = std::min(max_packet_size_bytes / aligned_page_size_bytes, num_pages);
        packet_size_bytes = aligned_page_size_bytes * max_num_pages_per_packet;
        total_packets = tt::div_up(num_pages, max_num_pages_per_packet);
    } else {
        max_num_pages_per_packet = 1;
        num_page_segments = tt::div_up(aligned_page_size_bytes, max_packet_size_bytes);
        packet_size_bytes = max_packet_size_bytes;
        total_packets = num_page_segments * num_pages;
    }

    return {packet_size_bytes, max_num_pages_per_packet, num_page_segments, total_packets};
}

auto fabric_1d_routing_vector(const MeshCoordinate& sender_coord, const MeshCoordinate& receiver_coord) {
    // transmit along row
    if (sender_coord[0] == receiver_coord[0]) {
        constexpr auto dim = 1;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    // transmit along col
    if (sender_coord[1] == receiver_coord[1]) {
        constexpr auto dim = 0;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    TT_THROW("Routing coordinates {} and {} invalid for 1D fabric", sender_coord, receiver_coord);
    return std::make_tuple(0, false, 0);
}

Fabric1DRoute fabric_1d_routing(
    const MeshDevice* mesh_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    const ::ttnn::ccl::Topology topology) {
    const auto& mesh_shape = mesh_device->get_view().shape();

    // sign indicates direction, however fabrics' forward/backward concept is reversed
    const auto [line_hops, line_is_forward, dim] = fabric_1d_routing_vector(sender_coord, receiver_coord);

    TT_FATAL(line_hops != 0, "Should not be send/receiving to the same device");

    auto get_neighbor_id = [&sender_coord, &mesh_device, &mesh_shape, dim](
                               bool is_forward, MeshCoordinate::BoundaryMode boundary_mode) {
        const auto neighbor_coord = sender_coord.get_neighbor(mesh_shape, (is_forward ? 1 : -1), dim, boundary_mode);

        TT_FATAL(neighbor_coord.has_value(), "Can't find neighbor for {}", sender_coord);
        return mesh_device->get_fabric_node_id(*neighbor_coord);
    };

    if (topology == ::ttnn::ccl::Topology::Ring) {
        int ring_hops = line_hops + ((line_hops < 0 ? -1 : 1) * mesh_shape[dim]);

        if (std::abs(ring_hops) < std::abs(line_hops)) {
            bool ring_is_forward = (ring_hops > 0);

            const auto next_fabric_id = get_neighbor_id(ring_is_forward, MeshCoordinate::BoundaryMode::WRAP);
            return {std::abs(ring_hops), !ring_is_forward, next_fabric_id};
        }
    }
    const auto next_fabric_id = get_neighbor_id(line_is_forward, MeshCoordinate::BoundaryMode::NONE);
    return {line_hops, !line_is_forward, next_fabric_id};
}
}  // namespace detail

void PointToPointOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(!input_tensor.is_sharded(), "Point to point does not yet support sharded configs");

    auto* mesh_device = input_tensor.device();

    // Same-device (send_coord == receive_coord) is allowed: it degenerates to a local
    // on-device copy of the shard into the output tensor — no fabric hop — handled by
    // the same-device branch in SendReceive::create_workload_descriptor.

    TT_FATAL(
        mesh_device->get_view().contains(operation_attributes.send_coord),
        "Mesh device must contain sender coordinate device");
    TT_FATAL(
        mesh_device->get_view().contains(operation_attributes.receive_coord),
        "Mesh device must contain receiver coordinate device");

    const auto& optional_output_tensor = tensor_args.optional_output_tensor;
    if (optional_output_tensor.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args).at(1);
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(output_tensor.layout() == input_tensor.layout(), "Output tensor must have same layout as input");

        TT_FATAL(
            output_spec == output_tensor.tensor_spec(),
            "Optional sparse output token tensor spec {} does not match computed output spec {}",
            output_tensor.tensor_spec(),
            output_spec);

        TT_FATAL(
            output_tensor.device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");
    }
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be 16 byte aligned");
};

PointToPointOp::spec_return_value_t PointToPointOp::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // !Maybe todo. Support output with different config/layout than input

    const auto& input_tensor = tensor_args.input_tensor;

    const auto final_output_spec = input_tensor.tensor_spec();

    // Same-device transfer is a local copy with no fabric packetization, so the
    // intermediate tensor is unused. Return a minimal 1-tile placeholder for it (rather
    // than a full input-sized, mesh-wide allocation) and skip compute_aligned_packet_dims —
    // its fabric query (get_tt_fabric_channel_buffer_size_bytes) requires an initialized
    // fabric context, which a purely local transfer must not depend on.
    if (operation_attributes.send_coord == operation_attributes.receive_coord) {
        const tt::tt_metal::TensorSpec placeholder_intermediate_spec(Shape{1, 1}, final_output_spec.tensor_layout());
        return {placeholder_intermediate_spec, final_output_spec};
    }

    const uint32_t input_num_pages = data_movement::get_num_pages(tensor_args.input_tensor);

    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            input_tensor.dtype(),
            final_output_spec.compute_page_size_bytes(),
            input_num_pages,
            ::hal::get_l1_alignment());

    const uint32_t packet_page_dim =
        packet_size_bytes / tt::datum_size(datatype_to_dataformat_converter(input_tensor.dtype()));

    Shape intermediate_shape{total_packets, packet_page_dim};

    tt::tt_metal::TensorSpec intermediate_spec(intermediate_shape, final_output_spec.tensor_layout());

    return {intermediate_spec, final_output_spec};
}

PointToPointOp::tensor_return_value_t PointToPointOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto* mesh_device = tensor_args.input_tensor.device();

    const auto intermediate_output_tensor =
        tensor_args.optional_intermediate_tensor.value_or(create_device_tensor(output_specs.at(0), mesh_device));

    const auto final_output_tensor =
        tensor_args.optional_output_tensor.value_or(create_device_tensor(output_specs.at(1), mesh_device));

    return {intermediate_output_tensor, final_output_tensor};
}

tt::tt_metal::WorkloadDescriptor PointToPointOp::SendReceive::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // Same-device transfer (send_coord == receive_coord) is a purely local on-device
    // copy: no fabric hop, so no cross-device GlobalSemaphore and no mesh-wide
    // Synchronize are needed. Emit a single copy program on that coordinate and return,
    // leaving the cross-device path below untouched.
    if (operation_attributes.send_coord == operation_attributes.receive_coord) {
        const MeshCoordinate& coord = operation_attributes.send_coord;
        const auto& coords = tensor_coords.coords();
        TT_FATAL(
            std::find(coords.begin(), coords.end(), coord) != coords.end(),
            "Tensor not present on coordinate: {}",
            coord);

        tt::tt_metal::WorkloadDescriptor workload_descriptor;
        workload_descriptor.programs.push_back(
            {tt::tt_metal::distributed::MeshCoordinateRange(coord),
             local_copy_program_factory(tensor_args, tensor_return_value)});
        return workload_descriptor;
    }

    auto* mesh_device = tensor_args.input_tensor.device();

    // Allocate the shared GlobalSemaphore used by both endpoint programs and
    // run the cross-device Synchronize barrier ONCE per workload (cache miss).
    // The semaphore's device-side allocation must outlive every program that
    // references its absolute address — park it in WorkloadDescriptor.semaphores
    // so the framework keeps it alive for the cached workload's lifetime.
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready in p2p op");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    log_debug(tt::LogOp, "Synchronize devices in p2p op done");

    const auto& send_coord = operation_attributes.send_coord;
    const auto& receive_coord = operation_attributes.receive_coord;

    const auto& coords = tensor_coords.coords();
    for (const auto& c : {send_coord, receive_coord}) {
        auto it = std::find(coords.begin(), coords.end(), c);
        TT_FATAL(it != coords.end(), "Tensor not present on coordinate: {}", c);
    }

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(semaphore);

    // Only the sender and receiver coords participate.  The original
    // create_mesh_workload likewise added programs only for these two coords;
    // we keep that behaviour by emitting per-coord PerCoordProgram entries.
    tt::tt_metal::ProgramDescriptor send_desc = send_program_factory(
        tensor_args, operation_attributes, send_coord, receive_coord, tensor_return_value, semaphore);
    workload_descriptor.programs.push_back(
        {tt::tt_metal::distributed::MeshCoordinateRange(send_coord), std::move(send_desc)});

    tt::tt_metal::ProgramDescriptor receive_desc =
        receive_program_factory(operation_attributes, tensor_return_value, semaphore);
    workload_descriptor.programs.push_back(
        {tt::tt_metal::distributed::MeshCoordinateRange(receive_coord), std::move(receive_desc)});

    return workload_descriptor;
}

}  // namespace ttnn::operations::point_to_point

namespace ttnn::prim {
ttnn::operations::point_to_point::PointToPointOp::tensor_return_value_t point_to_point(
    const Tensor& input_tensor,
    const ::ttnn::ccl::Topology& topology,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor) {
    using OperationType = ttnn::operations::point_to_point::PointToPointOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()},
        OperationType::tensor_args_t{input_tensor, optional_output_tensor, optional_intermediate_tensor});
}
}  // namespace ttnn::prim
