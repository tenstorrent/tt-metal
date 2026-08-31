// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ttnn/api/ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::kda_performance_model {

struct KdaFpuWork {
    uint64_t fpu_matrix_flops = 0;
    uint64_t fpu_multiply_ops = 0;
    uint64_t fpu_add_ops = 0;
    uint64_t fpu_reduction_ops = 0;
};

KdaFpuWork sigmoid_gated_rms_norm_work(uint64_t batch, uint64_t num_heads, uint64_t sequence, uint64_t value_dim);
KdaFpuWork qkv_causal_conv1d_silu_work(
    uint64_t batch, uint64_t sequence, uint64_t q_width, uint64_t k_width, uint64_t v_width);
KdaFpuWork reduce_affine_transforms_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim);
KdaFpuWork affine_exclusive_scan_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim);
KdaFpuWork prepare_chunk_recurrence_work(uint64_t num_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);
KdaFpuWork recurrent_chunk_scan_work(uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);
KdaFpuWork summarize_chunk_recurrence_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);

using KdaProfilerModel = tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>;

KdaProfilerModel make_profiler_model(
    const KdaFpuWork& work,
    std::span<const Tensor* const> inputs,
    const std::vector<Tensor>& outputs,
    tt::tt_metal::MathFidelity math_fidelity);

}  // namespace ttnn::experimental::prim::kda_performance_model
