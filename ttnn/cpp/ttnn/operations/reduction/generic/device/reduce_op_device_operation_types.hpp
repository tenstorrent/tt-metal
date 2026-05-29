// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "common.hpp"

namespace ttnn::prim {

struct ReduceParams {
    tt::tt_metal::ReduceOpMath math_op{};
    tt::tt_metal::ReduceOpDim dim{};
    float scaler{1.0f};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    bool negate{false};
    // For min/max with a non-unity scalar, the GMPOOL hardware path (reduce_tile LLK) only
    // respects the exponent of the scaler. To produce numerically correct results for any
    // scalar, the host instead requests reduction with `scaler=1.0` and applies the user
    // scalar afterwards via SFPU post-multiplication (mul_unary_tile) inside the compute
    // kernel, gated by the REDUCE_POST_MUL define. When `post_mul_scaler == 1.0f`, the
    // post-multiplication path is disabled and the existing reduce-only flow runs unchanged.
    float post_mul_scaler{1.0f};
};

}  // namespace ttnn::prim
