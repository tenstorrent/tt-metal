// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_pre_all_gather_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "layernorm_pre_all_gather_device_operation_types.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPreAllGatherDeviceOperation {
    using operation_attributes_t = LayerNormPreAllGatherOperationAttributes;
    using tensor_args_t = LayerNormPreAllGatherTensorArgs;
    using spec_return_value_t = LayerNormPreAllGatherSpecReturnValue;
    using tensor_return_value_t = LayerNormPreAllGatherTensorReturnValue;
    using program_factory_t = std::variant<
        program::LayerNormPreAllGatherProgramFactory,
        program::LayerNormPreAllGather2DProgramFactory,
        program::LayerNormPreAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        LayerNormDistributedType norm_type,
        tt::tt_metal::DataType dtype,
        const DeviceComputeKernelConfig& compute_kernel_config,
        const LayerNormProgramConfig& program_config,
        const std::optional<bool>& use_2d_core_grid);
};

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
constexpr auto layer_norm_pre_all_gather = ttnn::register_operation<
    "ttnn::prim::layer_norm_pre_all_gather",
    ttnn::operations::normalization::LayerNormPreAllGatherDeviceOperation>();
}  // namespace ttnn::prim
