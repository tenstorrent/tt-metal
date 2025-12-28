// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPreAllGatherOperationAttributes {
    LayerNormDistributedType norm_type;
    tt::tt_metal::DataType dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    LayerNormProgramConfig program_config;
    std::optional<bool> use_2d_core_grid;
};

struct LayerNormPreAllGatherTensorArgs {
    const Tensor& input;
};

using LayerNormPreAllGatherTensorReturnValue = Tensor;
using LayerNormPreAllGatherSpecReturnValue = TensorSpec;

}  // namespace ttnn::operations::normalization
