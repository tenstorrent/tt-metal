// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_post_all_gather_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "layernorm_post_all_gather_device_operation_types.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPostAllGatherDeviceOperation {
    using operation_attributes_t = LayerNormPostAllGatherOperationAttributes;
    using tensor_args_t = LayerNormPostAllGatherTensorArgs;
    using spec_return_value_t = LayerNormPostAllGatherSpecReturnValue;
    using tensor_return_value_t = LayerNormPostAllGatherTensorReturnValue;
    using program_factory_t = std::
        variant<program::LayerNormPostAllGatherProgramFactory, program::LayerNormPostAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const Tensor& stats,
        LayerNormDistributedType norm_type,
        float eps,
        const std::optional<const Tensor>& gamma,
        const std::optional<const Tensor>& beta,
        const MemoryConfig& memory_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        const std::optional<DataType>& dtype,
        const std::optional<bool>& use_2d_core_grid,
        const LayerNormProgramConfig& program_config);
};

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
constexpr auto layer_norm_post_all_gather = ttnn::register_operation<
    "ttnn::prim::layer_norm_post_all_gather",
    ttnn::operations::normalization::LayerNormPostAllGatherDeviceOperation>();
}  // namespace ttnn::prim
