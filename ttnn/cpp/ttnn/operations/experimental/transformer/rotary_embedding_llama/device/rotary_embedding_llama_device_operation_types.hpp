// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama {

struct RotaryEmbeddingLlamaParams {
    bool is_decode_mode{};
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct RotaryEmbeddingLlamaInputs {
    tt::tt_metal::Tensor input_tensor;
    tt::tt_metal::Tensor cos_cache;
    tt::tt_metal::Tensor sin_cache;
    tt::tt_metal::Tensor trans_mat;
};

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama
