// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_pre_all_gather_program_factory.hpp"
#include "ttnn/device_operation.hpp"

#include "layernorm_pre_all_gather_device_operation_types.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPreAllGatherDeviceOperation {
    using operation_attributes_t = LayerNormPreAllGatherOperationAttributes;
    using tensor_args_t = LayerNormPreAllGatherTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
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
        const std::optional<tt::tt_metal::DataType>& dtype,
        const DeviceComputeKernelConfig& compute_kernel_config,
        const LayerNormProgramConfig& program_config,
        const std::optional<bool>& use_2d_core_grid);
};

// Plain function to invoke the device operation
inline Tensor layer_norm_pre_all_gather(
    const Tensor& input,
    LayerNormDistributedType norm_type,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const LayerNormProgramConfig& program_config,
    const std::optional<bool>& use_2d_core_grid) {
    auto [operation_attributes, tensor_args] = LayerNormPreAllGatherDeviceOperation::invoke(
        input, norm_type, dtype, compute_kernel_config, program_config, use_2d_core_grid);
    return ttnn::device_operation::detail::invoke<LayerNormPreAllGatherDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::normalization
