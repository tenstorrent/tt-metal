// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

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

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    struct SingleCore {
        // Mesh-coord-aware overload: each program in the workload bakes in a per-device
        // weight-count offset derived from cluster_axis + mesh_coordinate, so the
        // descriptor framework dispatches one descriptor per coordinate.
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<SingleCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {};

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::MoeRoutingRemapDeviceOperation::tensor_return_value_t moe_routing_remap(
    const ttnn::Tensor& routing_weights,
    uint32_t non_zero_weight_size,
    uint32_t expert_parallel_size,
    uint32_t cluster_axis,
    const std::optional<ttnn::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_routing_weights = std::nullopt);
}  // namespace ttnn::prim
