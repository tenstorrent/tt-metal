// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_device_operation.hpp"

#include <initializer_list>
#include <tuple>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

namespace {
bool is_dram_interleaved(const ttnn::Tensor& t) {
    const auto& mem = t.memory_config();
    return mem.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}
}  // namespace

void UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    TT_FATAL(t.x.storage_type() == ttnn::StorageType::DEVICE, "x must be on device");
    // x layout/dtype depends on x_is_row_major:
    //   false (default): x is TILE BFLOAT8_B — the reader reads tile pages directly.
    //   true: x is ROW_MAJOR BFLOAT16 (the dispatch output) — the reader streams
    //     sticks and the compute kernel tilizes them to bf8_b before the matmul,
    //     fusing the standalone to_layout. Off preserves the pre-fusion path for
    //     standalone / Wormhole callers.
    if (op.x_is_row_major) {
        TT_FATAL(
            t.x.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "x must be BFLOAT16 when x_is_row_major, got {}",
            t.x.dtype());
        TT_FATAL(
            t.x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "x must be ROW_MAJOR when x_is_row_major, got {}",
            t.x.layout());
    } else {
        TT_FATAL(t.x.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "x must be BFLOAT8_B, got {}", t.x.dtype());
        TT_FATAL(t.x.layout() == tt::tt_metal::Layout::TILE, "x must be TILE layout");
    }
    TT_FATAL(is_dram_interleaved(t.x), "x must be DRAM-interleaved");
    TT_FATAL(t.x.logical_shape().rank() >= 2, "x must have rank >= 2, got rank {}", t.x.logical_shape().rank());
    // For rank > 2, all leading dims must be 1 — we treat x as effectively
    // (M, K) using padded_shape[-2:].
    for (int i = 0; i < static_cast<int>(t.x.logical_shape().rank()) - 2; ++i) {
        TT_FATAL(t.x.logical_shape()[i] == 1, "x leading dim {} must be 1, got {}", i, t.x.logical_shape()[i]);
    }

    // Per-local-expert weight lists: one entry per expert, all identical shape.
    // The kernels loop over experts and index a per-expert address array, so the
    // three lists must be non-empty, equal-length, and match experts_per_chip.
    TT_FATAL(
        !t.gate_projs.empty() && t.gate_projs.size() == t.up_projs.size() && t.gate_projs.size() == t.down_projs.size(),
        "gate/up/down projection lists must be non-empty and equal length (got {}, {}, {})",
        t.gate_projs.size(),
        t.up_projs.size(),
        t.down_projs.size());
    TT_FATAL(
        t.gate_projs.size() == op.experts_per_chip,
        "weight-list length ({}) must equal experts_per_chip ({})",
        t.gate_projs.size(),
        op.experts_per_chip);

    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_projs[0].padded_shape();
    const auto& up_shape = t.up_projs[0].padded_shape();
    const auto& down_shape = t.down_projs[0].padded_shape();

    TT_FATAL(
        x_shape[-1] == gate_shape[-2] && x_shape[-1] == up_shape[-2],
        "x's last dim {} must match gate/up's K dim ({}, {})",
        x_shape[-1],
        gate_shape[-2],
        up_shape[-2]);
    TT_FATAL(
        gate_shape[-1] == up_shape[-1] && gate_shape[-1] == down_shape[-2],
        "gate/up N ({}) must equal down K ({})",
        gate_shape[-1],
        down_shape[-2]);
    TT_FATAL(down_shape[-1] == x_shape[-1], "down N ({}) must equal x K ({})", down_shape[-1], x_shape[-1]);

    constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;
    TT_FATAL(x_shape[-2] % TILE == 0, "x M ({}) must be tile-aligned", x_shape[-2]);
    // m_tiles is this expert's M (grid/chunk/CB sizing). x may be a shared
    // buffer spanning many experts, so its allocated M only bounds m_tiles from
    // above — the reader/writer index into x at the region offset.
    TT_FATAL(op.m_tiles > 0, "m_tiles must be > 0");
    TT_FATAL(
        op.m_tiles <= x_shape[-2] / TILE, "m_tiles ({}) must be <= x M in tiles ({})", op.m_tiles, x_shape[-2] / TILE);

    // Every expert's gate/up/down tensor shares x's storage / layout / memory
    // contract AND must be identical in shape/dtype to expert 0 (the program is
    // built once for all experts, and the kernels reuse one accessor layout
    // descriptor per role with only the base address varying per expert).
    for (uint32_t e = 0; e < op.experts_per_chip; ++e) {
        for (const auto& [name, w, ref] :
             std::initializer_list<std::tuple<const char*, const ttnn::Tensor&, const ttnn::Tensor&>>{
                 {"gate_proj", t.gate_projs[e], t.gate_projs[0]},
                 {"up_proj", t.up_projs[e], t.up_projs[0]},
                 {"down_proj", t.down_projs[e], t.down_projs[0]}}) {
            TT_FATAL(w.storage_type() == ttnn::StorageType::DEVICE, "{}[{}] must be on device", name, e);
            TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "{}[{}] must be TILE layout", name, e);
            TT_FATAL(is_dram_interleaved(w), "{}[{}] must be DRAM-interleaved", name, e);
            TT_FATAL(
                w.padded_shape() == ref.padded_shape() && w.dtype() == ref.dtype(),
                "{}[{}] shape/dtype ({}, {}) must match expert 0 ({}, {}) — all experts share one program",
                name,
                e,
                w.padded_shape(),
                w.dtype(),
                ref.padded_shape(),
                ref.dtype());
        }
    }

    // Aux tensors: counts / global_expert_idx_table are small UINT32 vectors
    // the reader fetches via DRAM accessor. The reader does a single
    // noc_async_read_page(page=0, ...) and then indexes anywhere in
    // [0, num_global_experts), so the full vector must fit in one page. The
    // L1 scratch CB is sized to hold MAX_GLOBAL_EXPERTS UINT32 entries (see
    // the program factory), which covers DeepSeek V3 (256), Kimi (384) and any
    // model up to MAX_GLOBAL_EXPERTS routed experts. Validate the length here
    // so larger expert counts produce a clean assertion instead of silent OOB
    // reads at runtime.
    for (const auto& [name, a] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"counts", t.counts}, {"global_expert_idx_table", t.global_expert_idx_table}}) {
        TT_FATAL(a.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(a.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32", name);
        TT_FATAL(is_dram_interleaved(a), "{} must be DRAM-interleaved", name);
        const uint32_t num_entries = a.logical_shape()[-1];
        TT_FATAL(
            num_entries <= MAX_GLOBAL_EXPERTS,
            "{} length ({}) exceeds the maximum supported number of experts ({}) — "
            "the reader fetches only page 0 of this tensor into a fixed-size L1 scratch",
            name,
            num_entries,
            MAX_GLOBAL_EXPERTS);
    }
    TT_FATAL(
        op.experts_per_chip <= t.global_expert_idx_table.logical_shape()[-1],
        "experts_per_chip ({}) must be <= idx_table size ({})",
        op.experts_per_chip,
        t.global_expert_idx_table.logical_shape()[-1]);

    // The kernels always read each expert's x slice at its region offset
    // (fusing ttnn::extract) and write that expert's output into `output` at the
    // same offset (fusing ttnn::insert), so expert_region_offsets is mandatory.
    // This op just writes each expert's region into whatever `output` it was given.
    TT_FATAL(
        t.expert_region_offsets.has_value(),
        "expert_region_offsets is required (the kernels read/write per-expert regions)");
    {
        const auto& start = *t.expert_region_offsets;
        // These mirror ttnn::insert's validate_index_tensor for the `start`
        // tensor: by fusing insert into this op, the FFN now owns the
        // region-offset vector the writer fetches device-side, so it must
        // enforce the same invariants insert did. The writer does a single
        // noc_async_read_page(page 0) and indexes start[global_id], which is
        // only correct for a contiguous ROW_MAJOR single-page UINT32 vector.
        TT_FATAL(start.storage_type() == ttnn::StorageType::DEVICE, "expert_region_offsets must be on device");
        TT_FATAL(start.dtype() == tt::tt_metal::DataType::UINT32, "expert_region_offsets must be UINT32");
        TT_FATAL(
            start.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "expert_region_offsets must be ROW_MAJOR layout, got {}",
            start.layout());
        TT_FATAL(is_dram_interleaved(start), "expert_region_offsets must be DRAM-interleaved");
        const auto& start_shape = start.logical_shape();
        const bool start_valid_1d = start_shape.rank() == 1;
        const bool start_valid_2d = start_shape.rank() == 2 && start_shape[0] == 1;
        TT_FATAL(
            start_valid_1d || start_valid_2d,
            "expert_region_offsets must be 1D or 2D with first dimension == 1, got shape {}",
            start_shape);
        TT_FATAL(
            static_cast<uint32_t>(start_shape[-1]) <= MAX_GLOBAL_EXPERTS,
            "expert_region_offsets length ({}) exceeds the maximum supported number of experts ({})",
            start_shape[-1],
            MAX_GLOBAL_EXPERTS);
        // The writer reads start[global_id] and counts[global_id] from the same
        // global-expert index space, so the two vectors must be the same length
        // (mirrors ttnn::insert's start/counts last-dim check).
        TT_FATAL(
            start_shape[-1] == t.counts.logical_shape()[-1],
            "expert_region_offsets length ({}) must equal counts length ({})",
            start_shape[-1],
            t.counts.logical_shape()[-1]);
    }

    {
        const auto& out = t.output;
        TT_FATAL(out.storage_type() == ttnn::StorageType::DEVICE, "output must be on device");
        TT_FATAL(out.layout() == tt::tt_metal::Layout::TILE, "output must be TILE layout");
        TT_FATAL(is_dram_interleaved(out), "output must be DRAM-interleaved");
        // Output dtype must match x EXCEPT in row-major mode: there x is bf16
        // ROW_MAJOR but the tilized output is bf8_b TILE (for downstream
        // combine), so the two legitimately differ. The tilize/down-matmul packs
        // to the output's dtype regardless.
        TT_FATAL(
            op.x_is_row_major || out.dtype() == t.x.dtype(),
            "output dtype ({}) must match x dtype ({})",
            out.dtype(),
            t.x.dtype());
        const auto& out_shape = out.padded_shape();
        TT_FATAL(
            out_shape.rank() == x_shape.rank(),
            "output rank ({}) must match x rank ({})",
            out_shape.rank(),
            x_shape.rank());
        // Common to both modes: the N (emb) dim and all leading dims must match
        // x — the writer's tile-row stride is out_shape[-1]/TILE, and leading
        // dims index the same logical (1,..,1,M,N) tensor.
        TT_FATAL(
            out_shape[-1] == x_shape[-1],
            "output last dim ({}) must match x last dim ({})",
            out_shape[-1],
            x_shape[-1]);
        for (int i = 0; i < static_cast<int>(out_shape.rank()) - 2; ++i) {
            TT_FATAL(
                out_shape[i] == x_shape[i],
                "output leading dim {} ({}) must match x ({})",
                i,
                out_shape[i],
                x_shape[i]);
        }
        constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
        TT_FATAL(out_shape[-2] % TILE_H == 0, "output M ({}) must be tile-aligned", out_shape[-2]);
        TT_FATAL(out_shape[-2] >= x_shape[-2], "output M ({}) must be >= x M ({})", out_shape[-2], x_shape[-2]);
    }

    // Optional per-local-expert expert biases (gpt-oss). All-or-none: the three
    // lists are all empty or all length == experts_per_chip. gate/up bias last
    // dim == gate/up N (hidden); down bias last dim == down N (emb). Same device
    // / TILE / DRAM-interleaved contract as weights, and (like the weights) all
    // experts' biases share one shape/dtype.
    const int bias_lists = static_cast<int>(!t.gate_biases.empty()) + static_cast<int>(!t.up_biases.empty()) +
                           static_cast<int>(!t.down_biases.empty());
    TT_FATAL(
        bias_lists == 0 || bias_lists == 3,
        "gate/up/down bias lists must all be provided together or all omitted (got {} of 3)",
        bias_lists);
    const bool has_bias = bias_lists == 3;
    TT_FATAL(
        op.fuse_bias == has_bias, "op.fuse_bias ({}) must match presence of bias lists ({})", op.fuse_bias, has_bias);
    if (has_bias) {
        TT_FATAL(
            t.gate_biases.size() == op.experts_per_chip && t.up_biases.size() == op.experts_per_chip &&
                t.down_biases.size() == op.experts_per_chip,
            "each bias list must have experts_per_chip ({}) entries (got {}, {}, {})",
            op.experts_per_chip,
            t.gate_biases.size(),
            t.up_biases.size(),
            t.down_biases.size());
        for (uint32_t e = 0; e < op.experts_per_chip; ++e) {
            for (const auto& [name, b, expected_n] :
                 std::initializer_list<std::tuple<const char*, const ttnn::Tensor&, uint32_t>>{
                     {"gate_bias", t.gate_biases[e], static_cast<uint32_t>(gate_shape[-1])},
                     {"up_bias", t.up_biases[e], static_cast<uint32_t>(up_shape[-1])},
                     {"down_bias", t.down_biases[e], static_cast<uint32_t>(down_shape[-1])}}) {
                TT_FATAL(b.storage_type() == ttnn::StorageType::DEVICE, "{}[{}] must be on device", name, e);
                TT_FATAL(b.layout() == tt::tt_metal::Layout::TILE, "{}[{}] must be TILE layout", name, e);
                TT_FATAL(is_dram_interleaved(b), "{}[{}] must be DRAM-interleaved", name, e);
                // Exact LOGICAL shape: a single row of exactly `expected_n` columns. The
                // padded-width check below is necessary (the kernel/reader address tiles by
                // padded width) but not sufficient: shapes like (2, N) or (1, N-1) tile-pad
                // to the same width and would otherwise be accepted and silently mis-applied
                // (the reader loads only tile-row 0 and the compute kernel row-broadcasts it).
                const auto& lshape = b.logical_shape();
                TT_FATAL(
                    static_cast<uint32_t>(lshape[-1]) == expected_n && lshape.volume() == expected_n,
                    "{}[{}] logical shape {} must be a single row of its projection N ({})",
                    name,
                    e,
                    lshape,
                    expected_n);
                TT_FATAL(
                    static_cast<uint32_t>(b.padded_shape()[-1]) == expected_n,
                    "{}[{}] padded last dim ({}) must match its projection N ({})",
                    name,
                    e,
                    b.padded_shape()[-1],
                    expected_n);
            }
            // All bias CBs are configured from the gate-bias dtype (and the compute
            // kernel reuses one unpack format across gate/up), so every bias must share
            // a single dtype; a mixed-dtype call would read the wrong byte counts/formats.
            TT_FATAL(
                t.up_biases[e].dtype() == t.gate_biases[0].dtype() &&
                    t.down_biases[e].dtype() == t.gate_biases[0].dtype() &&
                    t.gate_biases[e].dtype() == t.gate_biases[0].dtype(),
                "all gate/up/down biases must share one dtype");
        }
        // Bias fusion is implemented only for the SwiGLU-OAI activation (gpt-oss):
        // the kernel adds gate/up bias before the clamp and down bias after the
        // down matmul. The SiLU path has no bias branch.
        TT_FATAL(
            op.activation == RoutedExpertActivation::SwiGluOai,
            "unified_routed_expert_moe: expert biases are only supported with RoutedExpertActivation::SwiGluOai "
            "(got the SiLU path).");
    }
}

void UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

UnifiedRoutedExpertFfnDeviceOperation::spec_return_value_t UnifiedRoutedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output.tensor_spec();
}

UnifiedRoutedExpertFfnDeviceOperation::tensor_return_value_t
UnifiedRoutedExpertFfnDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t& t) {
    return t.output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

namespace ttnn::prim {

ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& output,
    uint32_t m_tiles,
    uint32_t experts_per_chip,
    bool x_is_row_major,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation activation,
    const std::vector<ttnn::Tensor>& gate_biases,
    const std::vector<ttnn::Tensor>& up_biases,
    const std::vector<ttnn::Tensor>& down_biases) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .m_tiles = m_tiles,
            .experts_per_chip = experts_per_chip,
            .x_is_row_major = x_is_row_major,
            .activation = activation,
            .fuse_bias = !gate_biases.empty(),
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .x = x,
            .gate_projs = gate_projs,
            .up_projs = up_projs,
            .down_projs = down_projs,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table,
            .output = output,
            .expert_region_offsets = expert_region_offsets,
            .gate_biases = gate_biases,
            .up_biases = up_biases,
            .down_biases = down_biases});
}

}  // namespace ttnn::prim
