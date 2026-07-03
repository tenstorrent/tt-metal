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
// single noc_async_read_page, then indexes counts[global_expert_id] for
// global_expert_id in [0, num_global_experts). The L1 scratch is sized to
// hold this many UINT32 entries — 1024 entries = 4 KB ("4 tiles" of 1 KB) —
// which covers DeepSeek V3 (256 experts), Kimi (384 experts) and any model up
// to 1024 routed experts with headroom. A single ROW_MAJOR DRAM page already
// holds the whole vector, so the read stays a single page fetch; bumping this
// past TILE_HW would additionally require widening the device-op validation
// below and re-checking the per-core L1 budget.
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

    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple("chunk_M_tiles", "local_expert_id");
    auto attribute_values() const { return std::forward_as_tuple(chunk_M_tiles, local_expert_id); }
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
