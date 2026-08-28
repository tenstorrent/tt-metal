// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "ttnn/api/ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::kda_performance_model {

struct KdaWork {
    uint64_t dense_flops = 0;
    uint64_t multiply_results = 0;
    uint64_t add_results = 0;
    uint64_t reduction_input_elements = 0;
    uint64_t omitted_sfpu_results = 0;
};

struct KdaTensorTraffic {
    uint64_t buffer_address = 0;
    uint64_t physical_bytes = 0;
    bool is_dram = false;
};

struct KdaEstimate {
    bool valid = false;
    uint64_t ideal_fpu_cycles = 0;
    uint64_t ideal_fpu_ns = 0;
    uint64_t mandatory_dram_bytes = 0;
    uint64_t ideal_dram_ns = 0;
    uint64_t ideal_ns = 0;
    uint64_t omitted_sfpu_results = 0;
    std::vector<uint64_t> input_bytes;
    std::vector<uint64_t> output_bytes;
};

std::optional<KdaWork> sigmoid_gated_rms_norm_work(
    uint64_t batch, uint64_t num_heads, uint64_t sequence, uint64_t value_dim);
std::optional<KdaWork> qkv_causal_conv1d_silu_work(
    uint64_t batch, uint64_t sequence, uint64_t q_width, uint64_t k_width, uint64_t v_width);
std::optional<KdaWork> reduce_affine_transforms_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim);
std::optional<KdaWork> affine_exclusive_scan_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim);
std::optional<KdaWork> prepare_chunk_recurrence_work(
    uint64_t num_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);
std::optional<KdaWork> recurrent_chunk_scan_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);
std::optional<KdaWork> summarize_chunk_recurrence_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim);

std::optional<KdaTensorTraffic> tensor_traffic(const Tensor& tensor);
KdaEstimate zero_estimate(std::size_t input_count, std::size_t output_count);
KdaEstimate estimate(
    const KdaWork& work,
    std::span<const KdaTensorTraffic> inputs,
    std::span<const KdaTensorTraffic> outputs,
    uint64_t core_count,
    uint64_t clock_mhz,
    tt::tt_metal::MathFidelity math_fidelity);

template <typename OutputTensors>
tt::tt_metal::operation::OpPerformanceModelGeneral<OutputTensors> to_profiler_model(const KdaEstimate& estimate) {
    tt::tt_metal::operation::OpPerformanceModelGeneral<OutputTensors> result;
    result.inputs_bytes.assign(estimate.input_bytes.size(), 0);
    result.outputs_bytes.assign(estimate.output_bytes.size(), 0);
    if (!estimate.valid) {
        return result;
    }

    result.ideal_compute_cycles = std::max<int>(1, static_cast<int>(estimate.ideal_fpu_cycles));
    result.ideal_compute_ns = std::max<int>(1, static_cast<int>(estimate.ideal_fpu_ns));
    result.ideal_bandwidth_ns = std::max<int>(1, static_cast<int>(estimate.ideal_dram_ns));
    result.ideal_ns = std::max<int>(1, static_cast<int>(estimate.ideal_ns));
    std::transform(
        estimate.input_bytes.begin(), estimate.input_bytes.end(), result.inputs_bytes.begin(), [](uint64_t bytes) {
            return static_cast<int>(bytes);
        });
    std::transform(
        estimate.output_bytes.begin(), estimate.output_bytes.end(), result.outputs_bytes.begin(), [](uint64_t bytes) {
            return static_cast<int>(bytes);
        });
    return result;
}

}  // namespace ttnn::experimental::prim::kda_performance_model
