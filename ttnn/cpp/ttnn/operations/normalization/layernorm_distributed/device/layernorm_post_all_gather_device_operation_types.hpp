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
    // Optional: actual number of elements per device (for non-tile-aligned sizes)
    // When specified, this overrides the tensor width for computing the normalization factor.
    // This allows distributed norm to work correctly with padded tensors where
    // hidden_size_per_device is not tile-aligned.
    std::optional<uint32_t> num_elements_per_device;
};

struct LayerNormPostAllGatherInputs {
    const Tensor& input;
    const Tensor& stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

}  // namespace ttnn::prim
