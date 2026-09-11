// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "layernorm_device_operation_types.hpp"
#include "layernorm_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::prim {

// Complete L1-dependent identity of the interleaved program. Both the cache
// hash and factory use this plan so allocator changes matter only when they
// select a different descriptor.
struct LayerNormInterleavedPlan {
    bool use_welford = false;
    bool large_tensor = false;
    bool compact_fp32_finalizer = false;
    bool fused_pre_add_replay = false;
    bool affine_mcast = false;
    std::uint32_t width_block_tiles = 0;
    std::uint32_t input_tiles = 0;
    std::uint32_t residual_tiles = 0;
    std::uint32_t output_tiles = 0;
    std::uint32_t centred_tiles = 0;
    std::uint32_t squared_tiles = 0;
    std::uint32_t gamma_tiles = 0;
    std::uint32_t beta_tiles = 0;
    std::uint32_t residual_value_tiles = 0;
};

struct LayerNormMultiCoreProgramFactory {
    // The framework calls this with three arguments. The fourth restricts the cores the program may
    // touch: non-sharded layernorm splits its tile rows over whichever range it is given, so the
    // parameter selects the work-split grid, defaulting to the device's whole compute grid.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);

    // Returns the core range non-sharded LayerNorm distributes its tile rows over by default
    static CoreRangeSet default_core_range(tt::tt_metal::IDevice* device);

    static LayerNormInterleavedPlan select_plan(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

struct LayerNormShardedProgramFactory {
    // The framework calls this with three arguments. The fourth restricts the cores the program may
    // touch: sharded layernorm derives its cores from the input's shard spec, so the parameter only
    // validates that the multicast bounding box of that shard grid lies inside the given range.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

struct LayerNormDeviceOperation {
    using operation_attributes_t = LayerNormParams;
    using tensor_args_t = LayerNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
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
