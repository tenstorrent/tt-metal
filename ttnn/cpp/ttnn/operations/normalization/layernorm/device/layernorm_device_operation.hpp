// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

#include "layernorm_device_operation_types.hpp"
#include "layernorm_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::prim {

struct LayerNormMultiCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);

    // Returns the default core range for non-sharded LayerNorm if a
    // core range override is not provided
    static CoreRangeSet default_core_range(tt::tt_metal::IDevice* device);
};

struct LayerNormShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
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

    // Opts the op into the descriptor fast-path (no create_descriptor() rebuild on a cache hit).
    // Both factories bind every per-dispatch tensor address as a CB `.buffer` or Buffer* rt-arg
    // (the non-sharded factory binds input/gamma/beta/residual/output; the sharded factory binds
    // input/residual/stats/recip/output via CB and gamma/beta via Buffer* rt-args). Every other
    // runtime arg is shape/attr-derived and covered by the program hash, so there is nothing to
    // re-apply here.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
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
