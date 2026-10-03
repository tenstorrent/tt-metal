// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    // When set (interleaved + FUSE_PRE_ADD only), also write the pre-add sum (input + residual) to
    // residual_output_tensor — lets a resnet block fuse its terminal add into the next block's norm.
    bool output_residual_sum = false;
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> stats;                  // for POST_ALL_GATHER
    std::optional<Tensor> recip_tensor;           // reciprocal LUT for welford algorithm
    std::optional<Tensor> residual_output_tensor;  // preallocated pre-add sum output (output_residual_sum)
};

}  // namespace ttnn::prim
