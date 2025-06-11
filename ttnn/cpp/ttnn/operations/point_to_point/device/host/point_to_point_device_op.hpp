
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
        const MeshCoordinate& send_coord;
        const MeshCoordinate& receive_coord;
        const ccl::Topology topology;

        const tt::tt_metal::GlobalSemaphore receiver_semaphore;

        static constexpr auto attribute_names = std::forward_as_tuple("send_coord", "receive_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(send_coord, receive_coord, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
    };

    // entry 0 is the intermediate. Entry 1 is the final output
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    struct SendReceive {
        // !TODO
        struct shared_variables_t {};

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
            const ttnn::MeshCoordinate& send_coordinate,
            const ttnn::MeshCoordinate& receive_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // ! TODO
        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {};
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
        const ccl::Topology& topology,
        const MeshCoordinate& send_coord,
        const MeshCoordinate& receive_coord,
        const tt::tt_metal::GlobalSemaphore& receiver_semaphore) {
        return std::make_tuple(
            operation_attributes_t{send_coord, receive_coord, topology, receiver_semaphore},
            tensor_args_t{input_tensor});
    };

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> compute_aligned_packet_dims(
    const DataType& dtype, const uint32_t page_size_bytes, const uint32_t num_pages, const uint32_t alignment);

device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensor);

device_operation::CachedProgram<PointToPointOp::SendReceive::shared_variables_t> receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensor);

}  // namespace detail
}  // namespace operations::point_to_point

namespace prim {
constexpr auto point_to_point =
    ttnn::register_operation<"ttnn::prim::point_to_point", ttnn::operations::point_to_point::PointToPointOp>();
}  // namespace prim
}  // namespace ttnn
