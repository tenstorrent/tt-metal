// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct FusedRmsnormPreAllGatherParams {
    tt::tt_metal::DataType dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    // When num_heads > 1, the kernel emits `num_heads` stat tiles per row instead of 1
    // (one sum-of-squares stat per head's contiguous head_dim slice). Used by callers
    // who want per-head normalization downstream. Default 1 preserves the original
    // global-RMS behavior.
    uint32_t num_heads = 1;
};

struct FusedRmsnormPreAllGatherInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::experimental::prim
