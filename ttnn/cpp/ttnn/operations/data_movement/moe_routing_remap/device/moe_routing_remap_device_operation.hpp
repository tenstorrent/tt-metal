// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement {

struct MoeRoutingRemapDeviceOperation {
    struct operation_attributes_t {
        const uint32_t non_zero_weight_size;
        const uint32_t expert_parallel_size;
        const uint32_t cluster_axis;
        const std::optional<MemoryConfig> output_mem_config;

        static constexpr auto attribute_names =
            std::forward_as_tuple("non_zero_weight_size", "expert_parallel_size", "cluster_axis", "output_mem_config");
        auto attribute_values() const {
            return std::forward_as_tuple(non_zero_weight_size, expert_parallel_size, cluster_axis, output_mem_config);
        };
    };
    struct tensor_args_t {
        const ttnn::Tensor input_routing_weights;
        const std::optional<ttnn::Tensor> optional_output_routing_weights;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct SingleCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            ttnn::CoreCoord utilized_core;
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
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {};

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& routing_weights,
        uint32_t non_zero_weight_size,
        uint32_t expert_parallel_size,
        uint32_t cluster_axis,
        const std::optional<ttnn::MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_routing_weights = std::nullopt);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto moe_routing_remap = ttnn::register_operation<
    "ttnn::prim::moe_routing_remap",
    ttnn::operations::data_movement::MoeRoutingRemapDeviceOperation>();
}  // namespace ttnn::prim
