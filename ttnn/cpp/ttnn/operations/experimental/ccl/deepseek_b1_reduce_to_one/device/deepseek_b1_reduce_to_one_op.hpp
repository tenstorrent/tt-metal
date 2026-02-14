// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
namespace operations::experimental::ccl {

struct DeepseekB1ReduceToOneOp {
    struct operation_attributes_t {
        const MeshCoordinate& root_coord;
        const MeshCoordinate& exit_coord;
        const tt::tt_fabric::Topology topology;

        const ttnn::TensorSpec _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("root_coord", "exit_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(root_coord, exit_coord, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        const std::optional<Tensor> optional_output_tensor;
        const std::optional<std::vector<Tensor>> optional_intermediate_tensors;  // 3 tensors for 3 reduction rounds
    };

    using spec_return_value_t = std::array<std::vector<ttnn::TensorSpec>, 2>;
    using tensor_return_value_t = std::array<std::vector<ttnn::Tensor>, 2>;

    struct DeepseekB1ReduceToOne {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle send_reader_kernel_id;
            tt::tt_metal::KernelHandle send_worker_writer_kernel_id;  // worker_writer for all shard cores
            tt::tt_metal::KernelHandle send_fabric_writer_kernel_id;  // fabric_writer for dedicated fabric cores
            std::vector<CoreCoord> cores;                             // all shard cores (workers)
            std::vector<CoreCoord> fabric_cores;                      // dedicated fabric writer cores

            tt::tt_metal::KernelHandle root1_reader_kernel_id;
            tt::tt_metal::KernelHandle root1_writer_kernel_id;

            tt::tt_metal::KernelHandle root2_reader_kernel_id;
            tt::tt_metal::KernelHandle root2_writer_kernel_id;

            tt::tt_metal::KernelHandle compute_kernel_id;

            std::vector<tt::tt_metal::GlobalSemaphore> semaphores;

            bool is_mesh_leaf_device;
            bool is_mesh_root3_device;
            bool is_mesh_root2_device;
            bool is_mesh_root1_device;

            // CB handles for dynamic address updates
            tt::tt_metal::CBHandle local_cb_handle;
            tt::tt_metal::CBHandle received_cb_r1_handle;
            tt::tt_metal::CBHandle received_cb_r2_handle;
            tt::tt_metal::CBHandle received_cb_r3_handle;
            tt::tt_metal::CBHandle output_cb_handle;
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

    using program_factory_t = std::variant<DeepseekB1ReduceToOne>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return DeepseekB1ReduceToOne{};
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

device_operation::CachedProgram<DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::shared_variables_t>
deepseek_b1_reduce_to_one_program_factory(
    const DeepseekB1ReduceToOneOp::tensor_args_t& tensor_args,
    const DeepseekB1ReduceToOneOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const MeshCoordinate& device_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    DeepseekB1ReduceToOneOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores);
}  // namespace operations::experimental::ccl

namespace prim {
ttnn::operations::experimental::ccl::DeepseekB1ReduceToOneOp::tensor_return_value_t deepseek_b1_reduce_to_one(
    const Tensor& input_tensor,
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<std::vector<Tensor>>& optional_intermediate_tensors = std::nullopt);
}  // namespace prim
}  // namespace ttnn
