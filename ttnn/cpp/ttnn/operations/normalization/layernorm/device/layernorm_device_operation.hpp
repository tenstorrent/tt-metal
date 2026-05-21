// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/metal2_artifacts.hpp"

#include "layernorm_device_operation_types.hpp"
#include "layernorm_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::prim {

struct LayerNormMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value);

    // Returns the default core range for non-sharded LayerNorm if a
    // core range override is not provided
    static CoreRangeSet default_core_range(tt::tt_metal::IDevice* device);
};

struct LayerNormShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value);
};

struct LayerNormDeviceOperation {
    using operation_attributes_t = LayerNormParams;
    using tensor_args_t = LayerNormInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const MemoryConfig& output_mem_config,
    const LayerNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype = std::nullopt,
    LayerNormType norm_type = LayerNormType::LAYERNORM,
    DistributedLayerNormStage distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED,
    const std::optional<const Tensor>& stats = std::nullopt,
    const std::optional<const Tensor>& recip_tensor = std::nullopt,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation = std::nullopt);

}  // namespace ttnn::prim
