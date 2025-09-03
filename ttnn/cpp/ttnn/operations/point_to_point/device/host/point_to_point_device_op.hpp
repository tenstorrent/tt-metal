// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

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
    };

    // entry 0 is the intermediate. Entry 1 is the final output
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    struct SendReceive {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle send_unary_reader_kernel_id;
            tt::tt_metal::KernelHandle send_unary_writer_kernel_id;
            std::vector<CoreCoord> sender_cores;

            tt::tt_metal::KernelHandle receive_unary_reader_kernel_id;
            tt::tt_metal::KernelHandle receive_unary_writer_kernel_id;
            std::vector<CoreCoord> receiver_cores;
            const tt::tt_metal::GlobalSemaphore semaphore;
        };

        // AdaptedCachedMeshWorkload this maps device coordinates to sets of shared variables.
        // CachedMeshWorkload has a common set for all devices.
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const tt::tt_metal::GlobalSemaphore& semaphore);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SendReceive>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return SendReceive{};
    };

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    // Probably the same as on cache miss
    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        ;
        validate(operation_attributes, tensor_args);
    };

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const ::ttnn::ccl::Topology& topology,
        const MeshCoordinate& receiver_coord,
        const MeshCoordinate& sender_coord,
        const std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return std::make_tuple(
            operation_attributes_t{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()},
            tensor_args_t{input_tensor, optional_output_tensor});
    };

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

device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);

device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);
}  // namespace operations::point_to_point

namespace prim {
constexpr auto point_to_point =
    ttnn::register_operation<"ttnn::prim::point_to_point", ttnn::operations::point_to_point::PointToPointOp>();
}  // namespace prim
}  // namespace ttnn
