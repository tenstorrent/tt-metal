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
    // Output dtype; std::nullopt keeps the input's. Only the interleaved prefill factory supports a different one.
    std::optional<tt::tt_metal::DataType> output_dtype;
};

struct RotaryEmbeddingLlamaInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
    ttnn::Tensor trans_mat;
    // Paired prefill rope: input_tensor and input_tensor_b are both rotated in place by one program.
    std::optional<ttnn::Tensor> input_tensor_b;
};

}  // namespace ttnn::experimental::prim
