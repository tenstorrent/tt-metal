// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Every attribute and input set the merged op deals in: one per carried half, plus its own.
//
// Declared in dependency order -- the shared worker grid first, then each half, then the merged
// op -- because both halves take their rectangle and their activation from what precedes them.
// RoutedExpertActivation itself is NOT redeclared here: it is included from the op the unified
// half was carried from, so there is exactly one such C++ type. A second declaration compiles
// but produces a distinct type with no nanobind caster, which fails at `import ttnn` rather than
// at the call.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {
// The worker rectangle both passes run on. Rows 0-1 are reserved for the combine op and one
// column goes to dispatch, so 11x8 starting at y=2 is the whole grid this op may take.
//
// These are also the shard grid of the L1 arena, which is why they are named once rather than
// written at each use: the arena is HEIGHT_SHARDED one row per core, so a shard grid that does
// not match the rectangle hands some core an arena it does not own -- and the kernels address
// their buffers by a common offset, so that reads as corruption rather than a fault.
inline constexpr uint32_t kOriginY = 2;
inline constexpr uint32_t kGridX = 11;
inline constexpr uint32_t kGridY = 8;
}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified {

// The worker rectangle this op always runs on, independent of the device grid. Fixed rather than
// derived: the K-axis split, the activated L1 multicast pattern and the padded per_core_N all
// assume it, and the program factory asserts the device is at least this large.
//
// Exported because a hybrid routed-expert forward has to hand moe_fused_swiglu the SAME grid --
// that op defaults to the full device grid instead -- or the two halves block differently and the
// measured token-count crossover between them stops applying. Read it, do not restate it.

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

// The activation enum is the op this half was carried from's own type, not a copy: it is bound
// to Python once, as ttnn.RoutedExpertActivation, and a redeclaration here would be a distinct C++
// type that no nanobind caster knows -- which fails at module import, not at call time.
using ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation;

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

    // Active-token band this op owns. An expert whose count falls outside it is dropped
    // like a zero count, which is how a hybrid dispatch splits the experts between this op
    // and moe_fused_swiglu over ONE shared counts vector -- no masked tensors, no host sync.
    // Compile-time in the kernels, so it belongs to the program-cache key.
    uint32_t min_active_tokens = 0;
    uint32_t max_active_tokens = std::numeric_limits<uint32_t>::max();

    // The worker rectangle this half runs on. Defaults to the whole grid at
    // the grid origin, which is what a standalone dispatch uses.
    //
    // The merged op overrides it, but gives BOTH halves the SAME rectangle: a static split of the
    // grid makes the halves run concurrently on fractions of it, which measured 3.4x slower than
    // running each in turn on all of it. Sharing the cores is what forces the L1 arena, the shared
    // semaphore block and the pass barrier.
    uint32_t grid_x = kGridX;
    uint32_t grid_y = kGridY;
    uint32_t origin_x = 0;
    uint32_t origin_y = 0;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "m_tiles",
        "experts_per_chip",
        "x_is_row_major",
        "activation",
        "fuse_bias",
        "min_active_tokens",
        "max_active_tokens",
        "grid_x",
        "grid_y",
        "origin_x",
        "origin_y");
    auto attribute_values() const {
        return std::forward_as_tuple(
            m_tiles,
            experts_per_chip,
            x_is_row_major,
            activation,
            fuse_bias,
            min_active_tokens,
            max_active_tokens,
            grid_x,
            grid_y,
            origin_x,
            origin_y);
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused {

using unified::RoutedExpertActivation;

struct OperationArguments {
    uint32_t experts_per_chip = 1;
    uint32_t m_tiles = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    // Where this half's rectangle starts on the worker grid. The merged op gives BOTH halves the
    // same full rectangle and runs them one after the other, so origin_y skips the rows reserved
    // for combine rather than separating the halves. origin_x is 0 at every call site today.
    uint32_t origin_x = 0;
    uint32_t origin_y = 0;
    bool read_x_at_offset = false;
    // Active-token band this op owns. An expert whose count falls outside [min, max] is
    // dropped like a zero count -- uniform across the grid, so it costs no CB traffic, no
    // collective round and no semaphore. Wide open by default; a hybrid dispatch narrows it
    // so this op and unified_routed_expert_moe split the experts by load over ONE shared
    // counts vector, with no masked tensors and no host sync.
    uint32_t min_active_tokens = 0;
    uint32_t max_active_tokens = std::numeric_limits<uint32_t>::max();
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    // Whether the per-expert gate/up/down biases are fused. Derived from the presence of the bias
    // tensors, and hashed below so a biased and a bias-free dispatch of the same shape cannot share
    // a cached program -- the kernels select the bias adds with a compile-time FUSE_BIAS define.
    bool fuse_bias = false;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::MemoryConfig output_memory_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "experts_per_chip",
        "m_tiles",
        "grid_x",
        "grid_y",
        "origin_x",
        "origin_y",
        "read_x_at_offset",
        "min_active_tokens",
        "max_active_tokens",
        "activation",
        "fuse_bias",
        "output_dtype",
        "output_memory_config",
        "compute_kernel_config");

    auto attribute_values() const {
        return std::forward_as_tuple(
            experts_per_chip,
            m_tiles,
            grid_x,
            grid_y,
            origin_x,
            origin_y,
            read_x_at_offset,
            min_active_tokens,
            max_active_tokens,
            activation,
            fuse_bias,
            output_dtype,
            output_memory_config,
            compute_kernel_config);
    }
};

struct TensorArguments {
    Tensor activations;
    // One weight tensor per local expert. Expert 0 is the layout representative:
    // the program is built once and the kernels reuse a single accessor layout
    // descriptor per role, varying only the per-expert base address.
    std::vector<Tensor> w_gates;
    std::vector<Tensor> w_ups;
    std::vector<Tensor> w_downs;
    // Optional per-local-expert projection biases (gpt-oss). Either all three lists are populated
    // -- one bias per local expert, same length and order as the weight lists -- or all three are
    // empty: a partially biased dispatch would bias some projections and not others, which is wrong
    // numbers with no error. gate/up bias is (1, hidden), down is (1, emb).
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;
    Tensor counts;
    Tensor global_expert_idx_table;
    std::optional<Tensor> optional_output;
    std::optional<Tensor> expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

using unified::RoutedExpertActivation;

struct HybridRoutedExpertFfnParams {
    // Shared shape. Both halves are built for the same per-expert M and the same local expert
    // count; x is the wider shared dispatched buffer and each expert is placed at its region
    // offset, exactly as in either op alone.
    uint32_t m_tiles = 0;
    uint32_t experts_per_chip = 1;
    bool x_is_row_major = false;
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    bool fuse_bias = false;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    // The measured per-model split, read from the device-resident counts vector: an expert with
    // `count <= hybrid_token_threshold` runs on the fused half, the rest on the unified one. This
    // is the whole reason the two implementations are one op -- the choice is per expert and the
    // counts are only known on device, so neither half can be skipped host-side.
    //
    // Zero means no expert reaches the fused half: the unified half owns the layer, the fused
    // pass is not run and its circular buffers are not placed. Both bodies are still compiled in,
    // so the merged binaries are the same shape either way.
    uint32_t hybrid_token_threshold = 0;

    uint32_t origin_y = kOriginY;
    uint32_t grid_x = kGridX;
    uint32_t grid_y = kGridY;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "m_tiles",
        "experts_per_chip",
        "x_is_row_major",
        "activation",
        "fuse_bias",
        "compute_kernel_config",
        "hybrid_token_threshold",
        "origin_y",
        "grid_x",
        "grid_y");

    auto attribute_values() const {
        return std::forward_as_tuple(
            m_tiles,
            experts_per_chip,
            x_is_row_major,
            activation,
            fuse_bias,
            compute_kernel_config,
            hybrid_token_threshold,
            origin_y,
            grid_x,
            grid_y);
    }
};

// The union of both halves' inputs, which is just the unified half's list: the fused half consumes
// the same activations, weights, counts, index table, region offsets and biases, and writes into
// the same shared output.
struct HybridRoutedExpertFfnInputs {
    Tensor x;
    std::vector<Tensor> gate_projs;
    std::vector<Tensor> up_projs;
    std::vector<Tensor> down_projs;
    Tensor counts;
    Tensor global_expert_idx_table;
    Tensor output;
    std::optional<Tensor> expert_region_offsets;
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;

    // Per-core L1 scratch both halves' circular buffers are laid over. Required whenever pass A
    // runs: the two halves' buffers sum to more L1 than a core has, and overlaying them -- safe
    // because the passes are ordered, never concurrent -- is what lets both keep the whole grid.
    // Owned by the caller because the program keeps a raw pointer to it that must stay valid
    // across program-cache hits.
    std::optional<Tensor> l1_arena;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
