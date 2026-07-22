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

    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_proj.padded_shape();
    const auto& up_shape = t.up_proj.padded_shape();
    const auto& down_shape = t.down_proj.padded_shape();

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
    TT_FATAL(op.chunk_M_tiles > 0, "chunk_M_tiles must be > 0");
    // m_tiles is this expert's M (grid/chunk/CB sizing). x may be a shared
    // buffer spanning many experts, so its allocated M only bounds m_tiles from
    // above — the reader/writer index into x at the region offset.
    TT_FATAL(op.m_tiles > 0, "m_tiles must be > 0");
    TT_FATAL(
        op.m_tiles <= x_shape[-2] / TILE, "m_tiles ({}) must be <= x M in tiles ({})", op.m_tiles, x_shape[-2] / TILE);
    // read_x_at_offset needs expert_region_offsets to locate this expert's x
    // rows in the shared buffer (the reader fetches start[global_id]).
    TT_FATAL(
        !op.read_x_at_offset || t.expert_region_offsets.has_value(), "read_x_at_offset requires expert_region_offsets");

    // Weight tensors share x's storage / layout / memory contract — fail
    // host-side if the caller forgot to upload one, picked the wrong layout,
    // or sharded weights (the kernel reader assumes DRAM-interleaved).
    for (const auto& [name, w] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"gate_proj", t.gate_proj}, {"up_proj", t.up_proj}, {"down_proj", t.down_proj}}) {
        TT_FATAL(w.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
        TT_FATAL(is_dram_interleaved(w), "{} must be DRAM-interleaved", name);
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
        op.local_expert_id < t.global_expert_idx_table.logical_shape()[-1],
        "local_expert_id ({}) >= idx_table size ({})",
        op.local_expert_id,
        t.global_expert_idx_table.logical_shape()[-1]);

    // Direct-write mode: expert_region_offsets present => the writer places
    // this expert's output into the SHARED optional_output buffer at the
    // expert's region offset (fusing ttnn::insert). Requires optional_output.
    const bool direct_write = t.expert_region_offsets.has_value();
    if (direct_write) {
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
        TT_FATAL(
            t.optional_output.has_value(),
            "direct-write mode (expert_region_offsets set) requires optional_output (the shared destination buffer)");
    }

    if (t.optional_output.has_value()) {
        const auto& out = *t.optional_output;
        TT_FATAL(out.storage_type() == ttnn::StorageType::DEVICE, "optional_output must be on device");
        TT_FATAL(out.layout() == tt::tt_metal::Layout::TILE, "optional_output must be TILE layout");
        TT_FATAL(is_dram_interleaved(out), "optional_output must be DRAM-interleaved");
        // Output dtype must match x EXCEPT in row-major mode: there x is bf16
        // ROW_MAJOR but the tilized output is bf8_b TILE (for downstream
        // combine), so the two legitimately differ. The tilize/down-matmul packs
        // to the output's dtype regardless.
        TT_FATAL(
            op.x_is_row_major || out.dtype() == t.x.dtype(),
            "optional_output dtype ({}) must match x dtype ({})",
            out.dtype(),
            t.x.dtype());
        const auto& out_shape = out.padded_shape();
        TT_FATAL(
            out_shape.rank() == x_shape.rank(),
            "optional_output rank ({}) must match x rank ({})",
            out_shape.rank(),
            x_shape.rank());
        // Common to both modes: the N (emb) dim and all leading dims must match
        // x — the writer's tile-row stride is out_shape[-1]/TILE, and leading
        // dims index the same logical (1,..,1,M,N) tensor.
        TT_FATAL(
            out_shape[-1] == x_shape[-1],
            "optional_output last dim ({}) must match x last dim ({})",
            out_shape[-1],
            x_shape[-1]);
        for (int i = 0; i < static_cast<int>(out_shape.rank()) - 2; ++i) {
            TT_FATAL(
                out_shape[i] == x_shape[i],
                "optional_output leading dim {} ({}) must match x ({})",
                i,
                out_shape[i],
                x_shape[i]);
        }
        // Mode-specific M (row) dim: direct-write targets the larger shared
        // buffer (M >= x's M, tile-aligned; the writer bounds rows by
        // dst_M_tiles); otherwise the output is per-expert and M must match x.
        constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
        if (direct_write) {
            TT_FATAL(out_shape[-2] % TILE_H == 0, "optional_output M ({}) must be tile-aligned", out_shape[-2]);
            TT_FATAL(
                out_shape[-2] >= x_shape[-2],
                "optional_output M ({}) must be >= x M ({}) in direct-write mode",
                out_shape[-2],
                x_shape[-2]);
        } else {
            TT_FATAL(
                out_shape[-2] == x_shape[-2], "optional_output M ({}) must match x M ({})", out_shape[-2], x_shape[-2]);
        }
    }

    // Optional expert biases (gpt-oss). All-or-none: gate/up/down together or
    // none. gate/up bias last dim == gate/up N (hidden); down bias last dim ==
    // down N (emb). Same device / TILE / DRAM-interleaved contract as weights.
    const int bias_count = static_cast<int>(t.gate_bias.has_value()) + static_cast<int>(t.up_bias.has_value()) +
                           static_cast<int>(t.down_bias.has_value());
    TT_FATAL(
        bias_count == 0 || bias_count == 3,
        "gate/up/down biases must all be provided together or all omitted (got {} of 3)",
        bias_count);
    if (bias_count == 3) {
        for (const auto& [name, b, expected_n] :
             std::initializer_list<std::tuple<const char*, const ttnn::Tensor&, uint32_t>>{
                 {"gate_bias", *t.gate_bias, static_cast<uint32_t>(gate_shape[-1])},
                 {"up_bias", *t.up_bias, static_cast<uint32_t>(up_shape[-1])},
                 {"down_bias", *t.down_bias, static_cast<uint32_t>(down_shape[-1])}}) {
            TT_FATAL(b.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
            TT_FATAL(b.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
            TT_FATAL(is_dram_interleaved(b), "{} must be DRAM-interleaved", name);
            // Exact LOGICAL shape: a single row of exactly `expected_n` columns. The
            // padded-width check below is necessary (the kernel/reader address tiles by
            // padded width) but not sufficient: shapes like (2, N) or (1, N-1) tile-pad
            // to the same width and would otherwise be accepted and silently mis-applied
            // (the reader loads only tile-row 0 and the compute kernel row-broadcasts it).
            const auto& lshape = b.logical_shape();
            TT_FATAL(
                static_cast<uint32_t>(lshape[-1]) == expected_n && lshape.volume() == expected_n,
                "{} logical shape {} must be a single row of its projection N ({})",
                name,
                lshape,
                expected_n);
            TT_FATAL(
                static_cast<uint32_t>(b.padded_shape()[-1]) == expected_n,
                "{} padded last dim ({}) must match its projection N ({})",
                name,
                b.padded_shape()[-1],
                expected_n);
        }
        // All three bias CBs are configured from the gate-bias dtype (and the compute
        // kernel reuses one unpack format across gate/up), so the three biases must share
        // a single dtype; a mixed-dtype call would read the wrong byte counts/formats.
        TT_FATAL(
            t.up_bias->dtype() == t.gate_bias->dtype() && t.down_bias->dtype() == t.gate_bias->dtype(),
            "gate/up/down biases must share one dtype (got gate={}, up={}, down={})",
            t.gate_bias->dtype(),
            t.up_bias->dtype(),
            t.down_bias->dtype());
        // Bias fusion is implemented only for the SwiGLU-OAI activation (gpt-oss):
        // the kernel adds gate/up bias before the clamp and down bias after the
        // down matmul. The SiLU path has no bias branch.
        TT_FATAL(
            op.activation == RoutedExpertActivation::SwiGluOai,
            "unified_routed_expert_ffn: expert biases are only supported with RoutedExpertActivation::SwiGluOai "
            "(got the SiLU path).");
    }
}

void UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

UnifiedRoutedExpertFfnDeviceOperation::spec_return_value_t UnifiedRoutedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    if (t.optional_output.has_value()) {
        return t.optional_output->tensor_spec();
    }
    const ttnn::Shape output_shape(t.x.padded_shape());
    const auto mem =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(t.x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem));
}

UnifiedRoutedExpertFfnDeviceOperation::tensor_return_value_t
UnifiedRoutedExpertFfnDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& t) {
    if (t.optional_output.has_value()) {
        return *t.optional_output;
    }
    return create_device_tensor(compute_output_specs(op, t), t.x.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

namespace ttnn::prim {

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t chunk_M_tiles,
    uint32_t m_tiles,
    bool read_x_at_offset,
    bool x_is_row_major,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation activation,
    const std::optional<ttnn::Tensor>& gate_bias,
    const std::optional<ttnn::Tensor>& up_bias,
    const std::optional<ttnn::Tensor>& down_bias) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .chunk_M_tiles = chunk_M_tiles,
            .m_tiles = m_tiles,
            .local_expert_id = local_expert_id,
            .read_x_at_offset = read_x_at_offset,
            .x_is_row_major = x_is_row_major,
            .activation = activation,
            .fuse_bias = gate_bias.has_value(),
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .x = x,
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table,
            .optional_output = optional_output,
            .expert_region_offsets = expert_region_offsets,
            .gate_bias = gate_bias,
            .up_bias = up_bias,
            .down_bias = down_bias});
}

}  // namespace ttnn::prim
