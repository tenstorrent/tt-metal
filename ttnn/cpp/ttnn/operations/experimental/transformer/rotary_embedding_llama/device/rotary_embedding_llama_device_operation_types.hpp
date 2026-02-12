// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaParams {
    bool is_decode_mode{};
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("is_decode_mode", "output_mem_config", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(is_decode_mode, output_mem_config, compute_kernel_config);
    }
};

struct RotaryEmbeddingLlamaInputs {
    tt::tt_metal::Tensor input_tensor;
    tt::tt_metal::Tensor cos_cache;
    tt::tt_metal::Tensor sin_cache;
    tt::tt_metal::Tensor trans_mat;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensor", "cos_cache", "sin_cache", "trans_mat");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, cos_cache, sin_cache, trans_mat); }
};

}  // namespace ttnn::experimental::prim
