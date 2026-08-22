// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Maximum number of global experts the op supports.
//
// The reader fetches the per-global-expert `counts` vector (and the
// local->global `global_expert_idx_table`) into an L1 scratch CB with a
// single noc_async_read_page, then indexes counts[global_expert_id]. The CB is
// sized to the input tensor's aligned page; this constant caps the supported
// index space at 1024 UINT32 entries (4 KB), covering DeepSeek V3 (256), Kimi
// (384), and models up to 1024 routed experts. Raising it requires widening
// the device-op validation and re-checking the per-core L1 budget.
inline constexpr uint32_t MAX_GLOBAL_EXPERTS = tt::constants::TILE_HW;  // 1024

// Attributes (the constants known at host time).
struct UnifiedRoutedExpertFfnParams {
    // The compute kernel chunks the M axis into pieces of this many tiles so a
    // single matmul fits in per-core L1. 64 (= 2048 tokens) is the maximum that
    // keeps DeepSeek V3 routed-expert dims inside Blackhole L1.
    uint32_t chunk_M_tiles = 64;

    // Local expert id used to index `global_expert_idx_table` at runtime
    // (kernel reads global_id = idx_table[local_expert_id], then count =
    // counts[global_id]).
    uint32_t local_expert_id = 0;

    // When true, gate_proj/up_proj refer to the same packed tensor with
    // logical shape [1, local_experts, K, 2*N], and down_proj has shape
    // [1, local_experts, N, K]. The reader selects local_expert_id directly
    // from those stacked tensors and treats the first/second halves of the
    // packed last dimension as gate/up. This lets EP model weights stay in
    // their production representation instead of creating one Tensor handle
    // (and one duplicate allocation) per local expert.
    bool stacked_packed_weights = false;

    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    // compute_kernel_config affects the compiled kernel (at minimum fidelity
    // and approximation mode), so it must participate in the program-cache
    // key. Omitting it silently reused a LoFi program for a later HiFi call.
    static constexpr auto attribute_names =
        std::forward_as_tuple("chunk_M_tiles", "local_expert_id", "stacked_packed_weights", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(chunk_M_tiles, local_expert_id, stacked_packed_weights, compute_kernel_config);
    }
};

// Tensors fed into the op.
//
// x is the (M_max, K=emb) per-expert token buffer for this expert. Only the
// first `counts[global_expert_idx_table[local_expert_id]]` rows are valid;
// the rest is padding the FFN kernels must skip. Reader/writer always start
// at tile row 0 — the FFN op operates on an already-extracted per-expert
// tensor; a separate ttnn::extract / ttnn::insert pair handles slicing into
// / out of any shared dispatched buffer.
//
// gate_proj/up_proj/down_proj are the (K=emb, N=hidden), (K=emb, N=hidden),
// and (K=hidden, N=emb) weight tensors.
//
// counts/global_expert_idx_table are the device-side count buffers; the
// kernel reads them at runtime to skip unused chunks.
struct UnifiedRoutedExpertFfnInputs {
    Tensor x;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor counts;
    Tensor global_expert_idx_table;
    std::optional<Tensor> optional_output;
    // Direct-write mode: per-global-expert region start offsets (UINT32, the
    // same `start` tensor ttnn::insert consumes). When present, the writer
    // places this expert's output directly into `optional_output` (the shared
    // buffer) at start[global_id]/TILE tile-rows, fusing the ttnn::insert step.
    // Requires optional_output to also be set.
    std::optional<Tensor> expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
