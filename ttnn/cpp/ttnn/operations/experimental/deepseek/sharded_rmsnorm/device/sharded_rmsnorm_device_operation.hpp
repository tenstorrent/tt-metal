// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "sharded_rmsnorm_device_operation_types.hpp"
#include "sharded_rmsnorm_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm {

struct ShardedRmsNormDeviceOperation {
    using operation_attributes_t = ShardedRmsNormParams;
    using tensor_args_t = ShardedRmsNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ShardedRmsNormTensorReturn;
    using program_factory_t = std::variant<ShardedRmsNormProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm

namespace ttnn::prim {

Tensor sharded_rmsnorm(
    const Tensor& input_tensor,
    const Tensor& weight,
    float epsilon = 1e-6f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Layout> output_layout = std::nullopt);

}  // namespace ttnn::prim
