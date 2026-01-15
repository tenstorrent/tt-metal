// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::reduction::ema {

struct EmaParams {
    float alpha{};
    CoreCoord grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct EmaInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

}  // namespace ttnn::operations::reduction::ema
