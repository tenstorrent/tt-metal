// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineReduceParams {
    uint32_t expert_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeepseekMoEPostCombineReduceInputs {
    ttnn::Tensor combine_output;
    ttnn::Tensor weights;
};

using DeepseekMoEPostCombineReduceDeviceOperation = ttnn::device_operation::DeviceOperation<
    DeepseekMoEPostCombineReduceParams,
    DeepseekMoEPostCombineReduceInputs,
    ttnn::Tensor>;

class DeepseekMoEPostCombineReduceDeviceOperationImpl : public DeepseekMoEPostCombineReduceDeviceOperation {
public:
    using operation_attributes_t = DeepseekMoEPostCombineReduceParams;
    using tensor_args_t = DeepseekMoEPostCombineReduceInputs;
    using tensor_return_value_t = ttnn::Tensor;

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static ttnn::TensorSpec compute_output_specs(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static ttnn::Tensor create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static tt::stl::reflection::Attributes attributes();
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor deepseek_moe_post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    uint32_t expert_dim,
    const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim