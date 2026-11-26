// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

#include <vector>

namespace ttnn {

DeviceComputeKernelConfig rmsnorm_default_compute_config(tt::ARCH arch);

Tensor rms_norm(
    const Tensor& input_tensor,
    float epsilon = 1e-12,
    const std::optional<const Tensor>& weight = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    const std::optional<const Tensor>& residual_input_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const prim::LayerNormProgramConfig>& program_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt,
    // Preallocated interleaved TILE output: the result goes into its tile columns
    // [output_col_tiles, output_col_tiles + input width in tiles); its dtype is the output dtype.
    const std::optional<Tensor>& output_tensor = std::nullopt,
    uint32_t output_col_tiles = 0,
    // Up to two more tensors like output_tensor: the same result tiles are stored to each of them too.
    const std::vector<Tensor>& extra_output_tensors = {});

}  // namespace ttnn
