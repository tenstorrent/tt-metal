// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

struct LayerNormParams {
    LayerNormType norm_type = LayerNormType::LAYERNORM;
    DistributedLayerNormStage distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED;
    float eps = 0.0f;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    std::optional<operations::unary::UnaryWithParam> fused_activation;

    // Construct LayerNormParams with defaults resolved from input tensor context.
    // Optional parameters that are nullopt will be filled from the input tensor's
    // memory config, shard spec, tile dims, and device arch.
    // Compute kernel config defaults depend on norm_type (layernorm vs rmsnorm).
    static LayerNormParams with_defaults(
        const Tensor& input_tensor,
        LayerNormType norm_type,
        float eps,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        DistributedLayerNormStage distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<operations::unary::UnaryWithParam>& fused_activation = std::nullopt);
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> stats;                  // for POST_ALL_GATHER
    std::optional<Tensor> recip_tensor;           // reciprocal LUT for welford algorithm
};

}  // namespace ttnn::prim
