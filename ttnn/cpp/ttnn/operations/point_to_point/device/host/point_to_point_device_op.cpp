
#include <tt-metalium/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/fabric.hpp>

#include "point_to_point_device_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::point_to_point {

namespace detail {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> compute_aligned_packet_dims(
    const DataType& dtype, const uint32_t page_size_bytes, const uint32_t num_pages, const uint32_t alignment) {
    const auto fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
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

    return std::make_tuple(packet_size_bytes, max_num_pages_per_packet, num_page_segments, total_packets);
}
}  // namespace detail

using cached_program_t = device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t>;

void PointToPointOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(!input_tensor.is_sharded(), "Point to point does not yet support sharded configs");

    const uint32_t max_packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t page_size = tensor_args.input_tensor.tensor_spec().compute_page_size_bytes();
    TT_FATAL(page_size < max_packet_size, "Page size too large for P2P");

    auto input_device = dynamic_cast<MeshDevice*>(input_tensor.device());

    TT_FATAL(input_device != nullptr, "Point to point expected input tensor on mesh device");

    const auto mesh_device = operation_attributes.mesh_device();
    const auto&& input_device_ids = input_device->get_device_ids();
    TT_FATAL(input_device_ids.size() == 1, "Point to point expects input tensor MeshDevice of size 1");

    const auto&& output_device_ids = operation_attributes.receive_device->get_device_ids();
    TT_FATAL(input_device_ids.size() == 1, "Point to point expects output tensor MeshDevice of size 1");

    TT_FATAL(
        operation_attributes.send_coord != operation_attributes.receive_coord, "Can't send/receive to the same device");

    // ! TODO make sure this works with any MeshDevice where sender and receiver are subsets
    // currently let's restrict the MeshDevice to only contain sender/receiver, Maybe can lift that.
    const auto&& vmesh_device_ids = mesh_device->get_device_ids();
    const std::set<uint32_t> devices{input_device_ids.at(0), output_device_ids.at(0)},
        mesh_devices(vmesh_device_ids.begin(), vmesh_device_ids.end());

    TT_FATAL(devices == mesh_devices, "Mesh can only contain sender/receiver");
    TT_FATAL(devices.size() == 2, "point to point requires at least 2 devices");

    auto semaphore_device = dynamic_cast<MeshDevice*>(operation_attributes.receiver_semaphore.device());
    TT_FATAL(semaphore_device != nullptr, "Point to point expected semaphore on mesh device");

    const auto&& semaphore_device_ids = semaphore_device->get_device_ids();
    TT_FATAL(semaphore_device_ids.size() == 1, "Point to point expects semaphore MeshDevice of size 1");

    TT_FATAL(
        output_device_ids.at(0) == semaphore_device_ids.at(0), "Sempaphore must be associated with receiver device");
};

PointToPointOp::spec_return_value_t PointToPointOp::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // !Maybe todo. Support output with different config/layout than input

    const auto& input_tensor = tensor_args.input_tensor;

    const auto final_output_spec = input_tensor.tensor_spec();

    const uint32_t input_num_pages = data_movement::get_num_pages(tensor_args.input_tensor);

    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            input_tensor.get_dtype(),
            final_output_spec.compute_page_size_bytes(),
            input_num_pages,
            ::hal::get_l1_alignment());

    const uint32_t packet_page_dim =
        packet_size_bytes / tt::datum_size(datatype_to_dataformat_converter(input_tensor.get_dtype()));

    Shape intermediate_shape{total_packets, packet_page_dim};

    TensorSpec intermediate_spec(intermediate_shape, final_output_spec.tensor_layout());

    return {intermediate_spec, final_output_spec};
}

PointToPointOp::tensor_return_value_t PointToPointOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto dest_submesh_device = operation_attributes.receive_device;

    const auto intermediate_output_tensor = create_device_tensor(output_specs.at(0), dest_submesh_device);
    const auto final_output_tensor = create_device_tensor(output_specs.at(1), dest_submesh_device);

    return {intermediate_output_tensor, final_output_tensor};
}

PointToPointOp::SendReceive::cached_mesh_workload_t PointToPointOp::SendReceive::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    const auto send_coord = operation_attributes.send_coord;
    const auto receive_coord = operation_attributes.receive_coord;

    TT_ASSERT(tensor_coords.coords().size() == 2);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, send_coord, receive_coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

cached_program_t PointToPointOp::SendReceive::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const ttnn::MeshCoordinate& send_coordinate,
    const ttnn::MeshCoordinate& receive_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    if (mesh_coordinate == send_coordinate) {
        return detail::send_program_factory(
            tensor_args, operation_attributes, send_coordinate, receive_coordinate, tensor_return_value);

    } else if (mesh_coordinate == receive_coordinate) {
        return detail::receive_program_factory(operation_attributes, tensor_return_value);
    }

    TT_FATAL(true, "Bad coordinate in point_to_point");
    return cached_program_t{tt::tt_metal::Program{}, shared_variables_t{}};
}
}  // namespace ttnn::operations::point_to_point
