// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "layernorm_types.hpp"

namespace ttnn::operations::normalization::layer_norm {

struct LayerNormParams {
    LayerNormType norm_type = LayerNormType::LAYERNORM;
    DistributedLayerNormStage distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED;
    float eps = 0.0f;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> stats;                  // for POST_ALL_GATHER
};

}  // namespace ttnn::operations::normalization::layer_norm
