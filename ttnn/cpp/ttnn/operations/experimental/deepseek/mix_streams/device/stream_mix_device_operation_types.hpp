// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct StreamMixParams {
    uint32_t num_streams;  // hc, the streams' dim 1.
    uint32_t num_outputs;  // 1 (collapse with pre) or hc (residual mix with comb).
    MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct StreamMixInputs {
    const Tensor& streams;               // [1, hc, T, D], stream-major.
    const Tensor& weights;               // pre [1, T, 1, hc] or comb [1, T, hc, hc], one tile per token.
    std::optional<Tensor> post;          // [1, T, hc, 1], one tile per token; with sublayer_out only.
    std::optional<Tensor> sublayer_out;  // [1, 1, T, D].
};

using StreamMixTensorReturn = Tensor;  // [1, num_outputs, T, D]

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
