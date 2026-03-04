// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct LayerNormPostAllGatherParams {
    LayerNormDistributedType norm_type;
    float eps;
    tt::tt_metal::MemoryConfig memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> dtype;
    std::optional<bool> use_2d_core_grid;
    LayerNormProgramConfig program_config;
};

struct LayerNormPostAllGatherInputs {
    const Tensor& input;
    const Tensor& stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

}  // namespace ttnn::prim
