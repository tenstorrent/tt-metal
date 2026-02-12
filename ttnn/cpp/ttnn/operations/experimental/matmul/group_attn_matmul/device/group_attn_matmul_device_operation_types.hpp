// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct GroupAttnMatmulParams {
    std::optional<const uint32_t> num_tokens;
    std::optional<const bool> transpose_hw;
    const uint32_t out_subblock_w;
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    const bool row_major;  // Specifies how work is distributed across cores
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct GroupAttnMatmulInputs {
    Tensor input_tensor_a;
    Tensor input_tensor_b;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
