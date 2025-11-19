// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <tt_stl/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/global_semaphore.hpp"

#include "reduce_to_root_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::ccl {

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

auto fabric_routing_vector(const MeshCoordinate& sender_coord, const MeshCoordinate& receiver_coord) {
    // transmit along row
    if (sender_coord[0] == receiver_coord[0]) {
        constexpr auto dim = 1;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    // transmit along col
    else if (sender_coord[1] == receiver_coord[1]) {
        constexpr auto dim = 0;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);

        return std::make_tuple(std::abs(hops), is_fwd, dim);
    } else {
        TT_THROW("Routing coordinates {} and {} invalid for 1D fabric", sender_coord, receiver_coord);
        return std::make_tuple(0, false, 0);
    }
}

FabricRoute fabric_routing(
    const MeshDevice* mesh_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    const tt::tt_fabric::Topology topology) {
    const auto& mesh_shape = mesh_device->get_view().shape();

    // sign indicates direction, however fabrics' forward/backward concept is reversed
    const auto [line_hops, line_is_forward, dim] = fabric_routing_vector(sender_coord, receiver_coord);

    TT_FATAL(line_hops != 0, "Should not be send/receiving to the same device");

    auto get_neighbor_id = [&sender_coord, &mesh_device, &mesh_shape, dim](
                               bool is_forward, MeshCoordinate::BoundaryMode boundary_mode) {
        const auto neighbor_coord = sender_coord.get_neighbor(mesh_shape, (is_forward ? 1 : -1), dim, boundary_mode);

        TT_FATAL(neighbor_coord.has_value(), "Can't find neighbor for {}", sender_coord);
        return mesh_device->get_fabric_node_id(*neighbor_coord);
    };

    if (topology == tt::tt_fabric::Topology::Ring) {
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

using cached_workload_t = device_operation::CachedProgram<ReduceToRootOp::ReduceToRoot::shared_variables_t>;

void ReduceToRootOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_l = input_tensor[0];
    const auto& input_s = input_tensor[1];
    const auto& input_m = input_tensor[2];

    auto mesh_device = input_l.device();

    const auto& optional_output_tensor = tensor_args.optional_output_tensor;
    if (optional_output_tensor.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args).at(1);
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(output_tensor[0].layout() == input_l.layout(), "Output tensor must have same layout as input");
        TT_FATAL(output_tensor[1].layout() == input_s.layout(), "Output tensor must have same layout as input");
        TT_FATAL(output_tensor[2].layout() == input_m.layout(), "Output tensor must have same layout as input");

        for (size_t i = 0; i < output_tensor.size(); ++i) {
            TT_FATAL(
                output_spec[i] == output_tensor[i].tensor_spec(),
                "Optional sparse output token tensor spec {} does not match computed output spec {}",
                output_tensor[i].tensor_spec(),
                output_spec[i]);

            TT_FATAL(
                output_tensor[i].device() == mesh_device,
                "Output tensor must be allocated on same mesh device as input tensor");
        }
    }
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input_l.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be 16 byte aligned");
};

ReduceToRootOp::spec_return_value_t ReduceToRootOp::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // !Maybe todo. Support output with different config/layout than input

    const auto& input_tensor = tensor_args.input_tensor;

    std::vector<TensorSpec> final_output_spec = {
        input_tensor[0].tensor_spec(), input_tensor[1].tensor_spec(), input_tensor[2].tensor_spec()};

    std::vector<TensorSpec> intermediate_specs;
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t input_num_pages = data_movement::get_num_pages(tensor_args.input_tensor[i]);
        auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
            detail::compute_aligned_packet_dims(
                input_tensor[i].dtype(),
                final_output_spec[i].compute_page_size_bytes(),
                input_num_pages,
                ::hal::get_l1_alignment());

        uint32_t packet_page_dim =
            packet_size_bytes / tt::datum_size(datatype_to_dataformat_converter(input_tensor[i].dtype()));

        Shape intermediate_shape{total_packets, packet_page_dim};

        TensorSpec intermediate_spec(intermediate_shape, final_output_spec[i].tensor_layout());
        intermediate_specs.push_back(intermediate_spec);
    }

    return {intermediate_specs, final_output_spec};
}

ReduceToRootOp::tensor_return_value_t ReduceToRootOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto mesh_device = tensor_args.input_tensor[0].device();

    std::vector<ttnn::Tensor> intermediate_output_tensors;
    std::vector<ttnn::Tensor> final_output_tensors;
    for (size_t i = 0; i < 3; ++i) {
        auto intermediate_output_tensor = create_device_tensor(output_specs.at(0)[i], mesh_device);
        if (tensor_args.optional_intermediate_tensor.has_value()) {
            intermediate_output_tensor = tensor_args.optional_intermediate_tensor.value()[i];
        }

        auto final_output_tensor = create_device_tensor(output_specs.at(1)[i], mesh_device);
        if (tensor_args.optional_output_tensor.has_value()) {
            final_output_tensor = tensor_args.optional_output_tensor.value()[i];
        }
        intermediate_output_tensors.push_back(intermediate_output_tensor);
        final_output_tensors.push_back(final_output_tensor);
    }

    return {intermediate_output_tensors, final_output_tensors};
}

ReduceToRootOp::ReduceToRoot::cached_mesh_workload_t ReduceToRootOp::ReduceToRoot::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto mesh_device = tensor_args.input_tensor[0].device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    // 3 semaphores: first for devices 0,1, second for devices 2,3, third for devices 1 and 2
    for (size_t i = 0; i < 3; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    }
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready in reduce_to_root op");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    log_debug(tt::LogOp, "Synchronize devices in reduce_to_root op done");

    const auto& coords = tensor_coords.coords();

    for (const auto& coord : coords) {
        auto cached_workload = create_at(operation_attributes, coord, tensor_args, tensor_return_value, semaphores);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_workload.program));
        shared_variables.emplace(coord, std::move(cached_workload.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

cached_workload_t ReduceToRootOp::ReduceToRoot::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    const auto& root_coordinate = operation_attributes.root_coord;

    return reduce_to_root_program_factory(
        tensor_args, operation_attributes, root_coordinate, tensor_return_value, semaphores);

    return {Program{}, shared_variables_t{.semaphores = semaphores}};
}

}  // namespace ttnn::operations::ccl
