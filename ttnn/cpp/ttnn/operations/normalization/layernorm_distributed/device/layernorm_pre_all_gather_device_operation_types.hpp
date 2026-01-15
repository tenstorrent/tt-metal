// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization {

struct LayerNormPreAllGatherOperationAttributes {
    LayerNormDistributedType norm_type = LayerNormDistributedType::LAYERNORM;
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    ttnn::prim::LayerNormProgramConfig program_config;
    std::optional<bool> use_2d_core_grid;
};

struct LayerNormPreAllGatherTensorArgs {
    const Tensor& input;
};

}  // namespace ttnn::operations::normalization
