// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPostAllGatherOperationAttributes {
    LayerNormDistributedType norm_type;
    float eps;
    MemoryConfig memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    std::optional<bool> use_2d_core_grid;
    LayerNormProgramConfig program_config;
};

struct LayerNormPostAllGatherTensorArgs {
    const Tensor& input;
    const Tensor& stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

using LayerNormPostAllGatherTensorReturnValue = Tensor;
using LayerNormPostAllGatherSpecReturnValue = TensorSpec;

}  // namespace ttnn::operations::normalization
