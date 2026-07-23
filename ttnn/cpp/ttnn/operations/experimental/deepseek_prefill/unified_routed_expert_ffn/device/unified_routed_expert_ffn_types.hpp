// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

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

// Per-expert FFN activation variant. Selected at the op boundary and baked into
// the compute kernel via a compile-time define, so each variant caches as a
// distinct program. For SwiGluOai the alpha/limit are baked to the M3/gpt-oss
// values (1.702 / 7.0, SwiGLUConfigGPTOSS) in the kernel — no extra params.
enum class RoutedExpertActivation : uint8_t {
    Silu = 0,  // plain SiLU SwiGLU: silu(gate) * up                      (DeepSeek default)
    SwiGluOai =
        1,  // clamped swigluoai: (clamp(up,±L)+1)·clamp(gate,max=L)·σ(α·clamp(gate,max=L))  (MiniMax-M3 / gpt-oss)
};

// Attributes (the constants known at host time).
struct UnifiedRoutedExpertFfnParams {
    // Per-expert M dimension in tiles — the row count the matmul grid, chunk
    // loop, and CB sizes are built for. Every local expert shares this value
    // (= max_dispatched_tokens_per_expert / TILE), and x is the shared
    // dispatched buffer wider than one expert's region: the reader/writer index
    // into it at each expert's region offset while the op sizes its per-expert
    // work to this M.
    uint32_t m_tiles = 0;

    // Number of local experts this chip owns. The reader/compute/writer kernels
    // loop over local_expert in [0, experts_per_chip).
    uint32_t experts_per_chip = 1;

    // When true, x is a ROW_MAJOR bf16 buffer: the reader streams row-major
    // sticks and the compute kernel tilizes them (bf16 -> bf8_b) before the
    // gate/up matmul, fusing the standalone to_layout. False => x is already
    // TILE bf8_b (the reader reads tile pages directly).
    bool x_is_row_major = false;

    // Per-expert FFN activation variant. Baked into the compute kernel as a
    // compile-time define, so each variant caches as a distinct program — hence
    // it is part of the program-cache key below.
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;

    // Whether the (optional) gate/up/down expert biases are fused. Derived from
    // the presence of the bias tensors in the inputs. Drives a compile-time
    // FUSE_BIAS define in the compute/reader kernels, so a bias vs no-bias
    // program caches distinctly — hence it is part of the program-cache key.
    // (gpt-oss experts have gate/up/down biases; DeepSeek / MiniMax-M3 do not.)
    bool fuse_bias = false;

    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("m_tiles", "experts_per_chip", "x_is_row_major", "activation", "fuse_bias");
    auto attribute_values() const {
        return std::forward_as_tuple(m_tiles, experts_per_chip, x_is_row_major, activation, fuse_bias);
    }
};

// Tensors fed into the op.
//
// x is the (M_max, K=emb) shared dispatched buffer holding every local
// expert's tokens back to back. Expert `local_expert`'s rows begin at
// expert_region_offsets[global_id] and only its first counts[global_id] rows
// are valid; the kernels read each expert's slice at its region offset.
//
// gate_projs/up_projs/down_projs are per-local-expert weight lists (one entry
// per expert, all identical shape): (K=emb, N=hidden), (K=emb, N=hidden), and
// (K=hidden, N=emb). Each expert has its own DRAM buffer; the program factory
// passes every buffer's base address as a runtime-arg array and the kernels
// build a fresh accessor per expert from one shared layout descriptor.
//
// counts / global_expert_idx_table / expert_region_offsets are the device-side
// per-global-expert vectors; the kernels index them per local expert at
// runtime to size each expert's chunk loop and place its output.
//
// output is the shared destination buffer; each expert's result is written
// directly into its region at expert_region_offsets[global_id]/TILE tile-rows.
struct UnifiedRoutedExpertFfnInputs {
    Tensor x;
    std::vector<Tensor> gate_projs;
    std::vector<Tensor> up_projs;
    std::vector<Tensor> down_projs;
    Tensor counts;
    Tensor global_expert_idx_table;
    // Caller-provided shared destination buffer (always provided). Each expert
    // writes its region at expert_region_offsets[global_id]. Whether it aliases
    // x (in-place) is the caller's choice — the op just writes into it.
    Tensor output;
    // Per-global-expert region start offsets (UINT32, the same `start` tensor
    // ttnn::insert consumes). Required: the reader offsets each expert's x reads
    // and the writer places each expert's output at start[global_id]/TILE.
    std::optional<Tensor> expert_region_offsets;
    // Optional per-local-expert projection biases (gpt-oss). Either all three
    // lists are populated (one bias per local expert, same length/order as the
    // weight lists) or all three are empty. gate/up bias: (1, N=hidden); down
    // bias: (1, N=emb). When set, the fused kernel adds gate/up bias before the
    // activation and down bias after the down matmul. Empty for the bias-free
    // DeepSeek / MiniMax-M3 path (byte-identical).
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
