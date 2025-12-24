// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_pre_all_gather_program_factory.hpp"
#include "layernorm_pre_all_gather_2d_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "layernorm_pre_all_gather_device_operation_types.hpp"

namespace ttnn::operations::normalization::layernorm {

struct LayerNormPreAllGatherDeviceOperation {
    using operation_attributes_t = layernorm::operation_attributes_t;
    using tensor_args_t = layernorm::tensor_args_t;
    using spec_return_value_t = layernorm::spec_return_value_t;
    using tensor_return_value_t = layernorm::tensor_return_value_t;
    using program_factory_t =
        std::variant<program::LayerNormPreAllGatherProgramFactory, program::LayerNormPreAllGather2DProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::normalization::layernorm

namespace ttnn::prim {
ttnn::operations::normalization::layernorm::LayerNormPreAllGatherDeviceOperation::tensor_return_value_t
layernorm_pre_all_gather(
    const Tensor& input,
    ttnn::operations::normalization::LayerNormDistributedType norm_type,
    DataType dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<bool> use_2d_core_grid,
    const ttnn::operations::normalization::LayerNormDistributedDefaultProgramConfig& program_config,
    const std::optional<Tensor>& preallocated_output = std::nullopt);
}  // namespace ttnn::prim
