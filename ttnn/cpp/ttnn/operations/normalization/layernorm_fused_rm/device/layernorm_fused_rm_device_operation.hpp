// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_fused_rm_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "layernorm_fused_rm_device_operation_types.hpp"

namespace ttnn::operations::normalization::layernorm_fused_rm {

struct LayernormFusedRmDeviceOperation {
    using operation_attributes_t = LayernormFusedRmParams;
    using tensor_args_t = LayernormFusedRmInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::LayernormFusedRmProgramFactory>;

    // ALL STATIC FUNCTIONS - This is the modern pattern!
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::normalization::layernorm_fused_rm

namespace ttnn::prim {
// Primitive operation - free function that calls launch_on_device
ttnn::operations::normalization::layernorm_fused_rm::LayernormFusedRmDeviceOperation::tensor_return_value_t
layernorm_fused_rm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim
