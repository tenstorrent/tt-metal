// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

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
    // With LayerNormInputs::output_tensor: first tile column of the output's column window.
    uint32_t output_col_tiles = 0;
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> stats;                  // for POST_ALL_GATHER
    std::optional<Tensor> recip_tensor;           // reciprocal LUT for welford algorithm
    // Preallocated interleaved TILE output (same rows as the input, at least as wide): the result is written
    // into its tile columns [output_col_tiles, output_col_tiles + Wt); the other columns are left as they are.
    std::optional<Tensor> output_tensor;
    // Up to two more tensors with output_tensor's spec: the writer stores every result tile to each of them too.
    std::vector<Tensor> extra_output_tensors;
};

}  // namespace ttnn::prim
