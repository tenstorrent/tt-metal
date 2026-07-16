// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingParams {
    uint32_t seq_len = 0;
    std::optional<uint32_t> token_idx;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple("seq_len", "output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(seq_len, output_mem_config); }
};

struct RotaryEmbeddingInputs {
    Tensor input;
    Tensor cos;
    Tensor sin;
};

}  // namespace ttnn::experimental::prim
