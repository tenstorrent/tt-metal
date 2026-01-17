// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "layernorm_device_operation_types.hpp"
#include "layernorm_op_multi_core.hpp"
#include "layernorm_op_multi_core_sharded.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

struct LayerNormDeviceOperation {
    using operation_attributes_t = LayerNormParams;
    using tensor_args_t = LayerNormInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
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
    const std::optional<const Tensor>& stats = std::nullopt);

}  // namespace ttnn::prim
