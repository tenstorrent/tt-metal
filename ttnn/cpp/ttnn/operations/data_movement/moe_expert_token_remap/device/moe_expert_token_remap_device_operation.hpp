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

struct MoeExpertTokenRemapDeviceOperation {
    static constexpr uint32_t REDUCTION_SIZE = 16;

    struct operation_attributes_t {
        const std::optional<MemoryConfig> output_mem_config;
        const uint32_t reduction_size;

        static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config", "reduction_size");
        auto attribute_values() const { return std::forward_as_tuple(output_mem_config, reduction_size); };
    };
    struct tensor_args_t {
        const ttnn::Tensor topk_tensor;
        const ttnn::Tensor mapping_tensor;
        const ttnn::Tensor metadata_tensor;
        const std::optional<ttnn::Tensor> optional_output_mapping_tensor;
        const std::optional<ttnn::Tensor> optional_output_reduced_tensor;
    };

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;

    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    struct Multicore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<Multicore>;

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
ttnn::operations::data_movement::MoeExpertTokenRemapDeviceOperation::tensor_return_value_t moe_expert_token_remap(
    const ttnn::Tensor& topk_tensor,
    const ttnn::Tensor& mapping_tensor,
    const ttnn::Tensor& metadata_tensor,
    const std::optional<ttnn::MemoryConfig>& output_mem_config,
    const std::optional<ttnn::Tensor>& optional_output_mapping_tensor,
    const std::optional<ttnn::Tensor>& optional_output_reduced_tensor,
    uint32_t reduction_size = ttnn::operations::data_movement::MoeExpertTokenRemapDeviceOperation::REDUCTION_SIZE);
}  // namespace ttnn::prim
