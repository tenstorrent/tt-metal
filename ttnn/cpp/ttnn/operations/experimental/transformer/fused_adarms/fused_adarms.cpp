// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fused_adarms.hpp"

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt_stl/assert.hpp>

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> fused_adarms(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias,
    const ttnn::Tensor& cond,
    float epsilon,
    const std::optional<MemoryConfig>& memory_config) {
    const auto output_mem_cfg = memory_config.value_or(ttnn::L1_MEMORY_CONFIG);

    const uint32_t hidden_dim = input_tensor.logical_shape()[-1];
    const uint32_t batch_size = input_tensor.logical_shape()[0];

    // Linear projection: cond → modulation [batch, 1, hidden_dim * 3]
    auto modulation = ttnn::linear(
        cond,
        dense_weight,
        /*bias=*/dense_bias,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        output_mem_cfg);

    // Slice into scale, shift, gate — each [batch, 1, hidden_dim]
    ttnn::SmallVector<uint32_t> step{1u, 1u, 1u};
    auto scale = ttnn::slice<uint32_t>(
        modulation,
        ttnn::SmallVector<uint32_t>{0u, 0u, 0u},
        ttnn::SmallVector<uint32_t>{batch_size, 1u, hidden_dim},
        step);
    auto shift = ttnn::slice<uint32_t>(
        modulation,
        ttnn::SmallVector<uint32_t>{0u, 0u, hidden_dim},
        ttnn::SmallVector<uint32_t>{batch_size, 1u, hidden_dim * 2},
        step);
    auto gate = ttnn::slice<uint32_t>(
        modulation,
        ttnn::SmallVector<uint32_t>{0u, 0u, hidden_dim * 2},
        ttnn::SmallVector<uint32_t>{batch_size, 1u, hidden_dim * 3},
        step);
    modulation.deallocate();

    // RMSNorm with scale (pre-baked +1) as weight and shift as bias
    auto normed = ttnn::rms_norm(
        input_tensor,
        epsilon,
        scale,
        shift,
        /*residual_input_tensor=*/std::nullopt,
        output_mem_cfg);

    scale.deallocate();
    shift.deallocate();

    return {normed, gate};
}

}  // namespace ttnn::experimental
