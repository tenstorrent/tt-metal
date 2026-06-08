// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"

#include "point_to_point_device_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::point_to_point {

void PointToPointOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(!input_tensor.is_sharded(), "Point to point does not yet support sharded configs");

    auto* mesh_device = input_tensor.device();

    TT_FATAL(
        operation_attributes.send_coord != operation_attributes.receive_coord, "Can't send/receive to the same device");

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
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // !Maybe todo. Support output with different config/layout than input

    const auto& input_tensor = tensor_args.input_tensor;

    const auto final_output_spec = input_tensor.tensor_spec();

    const uint32_t input_num_pages = data_movement::get_num_pages(tensor_args.input_tensor);

    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        ::ttnn::ccl::dataflow::ccl_packet_dims(
            input_tensor.dtype(),
            final_output_spec.compute_page_size_bytes(),
            input_num_pages,
            ::hal::get_l1_alignment());

    const uint32_t packet_page_dim =
        packet_size_bytes / tt::datum_size(datatype_to_dataformat_converter(input_tensor.dtype()));

    Shape intermediate_shape{total_packets, packet_page_dim};

    TensorSpec intermediate_spec(intermediate_shape, final_output_spec.tensor_layout());

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
    auto* mesh_device = tensor_args.input_tensor.device();

    // Allocate the shared GlobalSemaphore used by both endpoint programs and
    // run the cross-device Synchronize barrier ONCE per workload (cache miss).
    // The semaphore's device-side allocation must outlive every program that
    // references its absolute address — park it in WorkloadDescriptor.semaphores
    // so the framework keeps it alive for the cached workload's lifetime.
    // Allocate the shared GlobalSemaphore and run the cache-miss cross-device Synchronize
    // barrier (helper owns both); the caller parks it in WorkloadDescriptor::semaphores below.
    auto semaphore = ::ttnn::ccl::dataflow::make_ccl_semaphore(mesh_device, 0);

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
