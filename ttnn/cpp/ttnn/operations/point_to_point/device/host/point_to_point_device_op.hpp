// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include "ttnn/operations/ccl/ccl_common.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {
namespace operations::point_to_point {

struct PointToPointOp {
    struct operation_attributes_t {
        const MeshCoordinate& receive_coord;
        const MeshCoordinate& send_coord;
        const ::ttnn::ccl::Topology topology;

        // put this in here to hash on tensor spec
        const ttnn::TensorSpec _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("send_coord", "receive_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(send_coord, receive_coord, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
        const std::optional<ttnn::Tensor> optional_intermediate_tensor;
    };

    // entry 0 is the intermediate. Entry 1 is the final output
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    struct SendReceive {
        // Builds the entire workload in one call (cache miss):
        //   1. Allocates the shared GlobalSemaphore used by both endpoint
        //      programs and runs the cross-device Synchronize barrier; the
        //      semaphore is parked in `WorkloadDescriptor::semaphores` so it
        //      outlives the cached workload.
        //   2. Builds ProgramDescriptors for the send_coord (via
        //      send_program_factory) and the receive_coord (via
        //      receive_program_factory) and pushes them as single-coord
        //      ranges into `programs`.  No program is emitted for any other
        //      mesh coord — matches the legacy create_mesh_workload that only
        //      added programs at the two endpoint coords.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<SendReceive>;

    // Mandatory methods

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

namespace detail {

struct AlignedPacketDims {
    const uint32_t packet_size_bytes;
    const uint32_t max_num_pages_per_packet;
    const uint32_t num_page_segments;
    const uint32_t total_packets;
};

AlignedPacketDims compute_aligned_packet_dims(
    const DataType& dtype, uint32_t page_size_bytes, uint32_t num_pages, uint32_t alignment);

struct Fabric1DRoute {
    const uint32_t num_hops;
    const bool is_forward;
    const tt::tt_fabric::FabricNodeId neighbor_id;
};

Fabric1DRoute fabric_1d_routing(
    const MeshDevice* mesh_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    ::ttnn::ccl::Topology topology);

}  // namespace detail

tt::tt_metal::ProgramDescriptor send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);

tt::tt_metal::ProgramDescriptor receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);

// Same-device (send_coord == receive_coord) local on-device copy — no fabric.
tt::tt_metal::ProgramDescriptor local_copy_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args, PointToPointOp::tensor_return_value_t& output_tensors);
}  // namespace operations::point_to_point

namespace prim {
ttnn::operations::point_to_point::PointToPointOp::tensor_return_value_t point_to_point(
    const Tensor& input_tensor,
    const ::ttnn::ccl::Topology& topology,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor = std::nullopt);
}  // namespace prim

namespace device_operation {
// `create_output_tensors` resolves `optional_output_tensor` -> `tensor_return_value[1]`
// and `optional_intermediate_tensor` -> `tensor_return_value[0]` (same Tensor object,
// same Buffer*).  Letting the default reflective walk enumerate the optionals AND the
// return slots produces duplicate `Buffer*`s in the buffer list, which trips the
// aliasing guard in `resolve_bindings` (program_descriptor_patching.cpp).  For
// contract-2 ops that guard makes the cache-hit fast path a no-op — buffer addresses
// are never re-patched and the program runs against stale L1/DRAM addresses from the
// first cache miss (issue #45422).
//
// Skip the optionals here: when set they alias entries in `tensor_return_value` and
// would only add duplicates; when unset they contribute nothing.  The return-value
// walk still enumerates the canonical intermediate/final buffers.
template <>
struct extract_tensor_buffers_t<::ttnn::operations::point_to_point::PointToPointOp::tensor_args_t, void> {
    template <typename Out>
    static void call(
        const ::ttnn::operations::point_to_point::PointToPointOp::tensor_args_t& args, Out& out) {
        out.push_back(args.input_tensor.buffer());
    }
};
}  // namespace device_operation
}  // namespace ttnn
