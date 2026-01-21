// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnMatmulParams {
    const std::optional<const uint32_t> num_tokens;
    const std::optional<const bool> transpose_hw;
    const CoreCoord compute_with_storage_grid_size;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct AttnMatmulInputs {
    Tensor input_tensor_a;
    Tensor input_tensor_b;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
