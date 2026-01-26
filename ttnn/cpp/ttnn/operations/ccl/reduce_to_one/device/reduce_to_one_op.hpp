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
namespace operations::ccl {

struct ReduceToOneOp {
    struct operation_attributes_t {
        const MeshCoordinate& root_coord;
        const tt::tt_fabric::Topology topology;

        const ttnn::TensorSpec _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("root_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(root_coord, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        const std::optional<Tensor> optional_output_tensor;
        const std::optional<Tensor> optional_intermediate_tensor;
    };

    using spec_return_value_t = std::array<std::vector<ttnn::TensorSpec>, 2>;
    using tensor_return_value_t = std::array<std::vector<ttnn::Tensor>, 2>;

    struct ReduceToOne {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle send_reader_kernel_id;
            tt::tt_metal::KernelHandle send_writer_kernel_id;
            std::vector<CoreCoord> cores;

            tt::tt_metal::KernelHandle root1_reader_kernel_id;
            tt::tt_metal::KernelHandle root1_writer_kernel_id;

            tt::tt_metal::KernelHandle root2_reader_kernel_id;
            tt::tt_metal::KernelHandle root2_writer_kernel_id;

            tt::tt_metal::KernelHandle compute_kernel_id;

            std::vector<tt::tt_metal::GlobalSemaphore> semaphores;

            bool is_mesh_leaf_device;
            bool is_root_device;
            bool is_mesh_root2_device;
            bool is_col_root_device;
        };

        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            std::optional<MeshCoordinate>& forward_coord,
            std::optional<MeshCoordinate>& backward_coord,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ReduceToOne>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return ReduceToOne{};
    };

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

device_operation::CachedProgram<ReduceToOneOp::ReduceToOne::shared_variables_t> reduce_to_one_program_factory(
    const ReduceToOneOp::tensor_args_t& tensor_args,
    const ReduceToOneOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& device_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    ReduceToOneOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);
}  // namespace operations::ccl

namespace prim {
ttnn::operations::ccl::ReduceToOneOp::tensor_return_value_t reduce_to_one(
    const Tensor& input_tensor,
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<Tensor>& optional_intermediate_tensor = std::nullopt);
}  // namespace prim
}  // namespace ttnn
