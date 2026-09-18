// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaParams {
    bool is_decode_mode{};
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct RotaryEmbeddingLlamaInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
    ttnn::Tensor trans_mat;
};

}  // namespace ttnn::experimental::prim
