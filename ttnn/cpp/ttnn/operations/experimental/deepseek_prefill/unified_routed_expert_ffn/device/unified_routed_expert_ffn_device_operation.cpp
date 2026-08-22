// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_device_operation.hpp"

#include <initializer_list>
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

void validate_index_tensor(const ttnn::Tensor& tensor, const char* name, const ttnn::Tensor& x) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.device() == x.device(), "{} must be on the same device as x", name);
    TT_FATAL(tensor.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "{} must be ROW_MAJOR layout, got {}",
        name,
        tensor.layout());
    TT_FATAL(is_dram_interleaved(tensor), "{} must be DRAM-interleaved", name);

    const auto& shape = tensor.logical_shape();
    const bool valid_1d = shape.rank() == 1;
    const bool valid_2d = shape.rank() == 2 && shape[0] == 1;
    TT_FATAL(valid_1d || valid_2d, "{} must be 1D or 2D with first dimension == 1, got shape {}", name, shape);
}
}  // namespace

void UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    TT_FATAL(t.x.storage_type() == tt::tt_metal::StorageType::DEVICE, "x must be on device");
    // BFLOAT8_B remains the DeepSeek production path. BFLOAT16 is also
    // supported for models whose activation accuracy cannot tolerate the
    // BFP8 dispatch/intermediate/output boundaries. The program factory uses
    // x's dtype for the intermediate CBs, so this is a real end-to-end BF16
    // path rather than a BF16 input immediately repacked to BFP8.
    TT_FATAL(
        t.x.dtype() == tt::tt_metal::DataType::BFLOAT8_B || t.x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "x must be BFLOAT8_B or BFLOAT16, got {}",
        t.x.dtype());
    TT_FATAL(t.x.layout() == tt::tt_metal::Layout::TILE, "x must be TILE layout");
    TT_FATAL(is_dram_interleaved(t.x), "x must be DRAM-interleaved");
    TT_FATAL(t.x.buffer() != nullptr, "x must have a buffer");
    TT_FATAL(t.x.logical_shape().rank() >= 2, "x must have rank >= 2, got rank {}", t.x.logical_shape().rank());
    // For rank > 2, all leading dims must be 1 — we treat x as effectively
    // (M, K) using padded_shape[-2:].
    for (int i = 0; i < static_cast<int>(t.x.logical_shape().rank()) - 2; ++i) {
        TT_FATAL(t.x.logical_shape()[i] == 1, "x leading dim {} must be 1, got {}", i, t.x.logical_shape()[i]);
    }

    // Weight tensors share x's storage / layout / memory contract. Validate
    // these before indexing their trailing dimensions below so malformed API
    // inputs fail deterministically on the host.
    for (const auto& [name, w] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"gate_proj", t.gate_proj}, {"up_proj", t.up_proj}, {"down_proj", t.down_proj}}) {
        TT_FATAL(w.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(w.buffer() != nullptr, "{} must have a buffer", name);
        TT_FATAL(w.device() == t.x.device(), "{} must be on the same device as x", name);
        TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
        TT_FATAL(is_dram_interleaved(w), "{} must be DRAM-interleaved", name);
        TT_FATAL(w.logical_shape().rank() >= 2, "{} must have rank >= 2, got rank {}", name, w.logical_shape().rank());
    }

    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_proj.padded_shape();
    const auto& up_shape = t.up_proj.padded_shape();
    const auto& down_shape = t.down_proj.padded_shape();

    if (op.stacked_packed_weights) {
        TT_FATAL(
            t.gate_proj.logical_shape().rank() == 4 && t.up_proj.logical_shape().rank() == 4 &&
                t.down_proj.logical_shape().rank() == 4,
            "stacked packed weights must have rank 4 (gate {}, up {}, down {})",
            t.gate_proj.logical_shape().rank(),
            t.up_proj.logical_shape().rank(),
            t.down_proj.logical_shape().rank());
        TT_FATAL(
            t.gate_proj.logical_shape()[0] == 1 && t.up_proj.logical_shape()[0] == 1 &&
                t.down_proj.logical_shape()[0] == 1,
            "stacked packed weights must have leading dimension 1 (gate {}, up {}, down {})",
            t.gate_proj.logical_shape()[0],
            t.up_proj.logical_shape()[0],
            t.down_proj.logical_shape()[0]);
        TT_FATAL(
            gate_shape == up_shape,
            "stacked gate/up views must have identical shapes, got {} and {}",
            gate_shape,
            up_shape);
        TT_FATAL(
            gate_shape[-3] == down_shape[-3],
            "stacked gate_up/down local-expert dimensions must match ({} vs {})",
            gate_shape[-3],
            down_shape[-3]);
        TT_FATAL(
            op.local_expert_id < gate_shape[-3],
            "local_expert_id ({}) >= stacked local expert count ({})",
            op.local_expert_id,
            gate_shape[-3]);
        TT_FATAL(
            gate_shape[-1] % 2 == 0,
            "stacked gate_up last dimension ({}) must split evenly into gate/up",
            gate_shape[-1]);
        TT_FATAL(
            x_shape[-1] == gate_shape[-2],
            "x's last dim {} must match stacked gate_up K dim {}",
            x_shape[-1],
            gate_shape[-2]);
        TT_FATAL(
            gate_shape[-1] / 2 == down_shape[-2],
            "half of stacked gate_up N ({}) must equal down K ({})",
            gate_shape[-1] / 2,
            down_shape[-2]);
        TT_FATAL(down_shape[-1] == x_shape[-1], "stacked down N ({}) must equal x K ({})", down_shape[-1], x_shape[-1]);
    } else {
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
    }

    constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;
    TT_FATAL(x_shape[-2] % TILE == 0, "x M ({}) must be tile-aligned", x_shape[-2]);
    TT_FATAL(op.chunk_M_tiles > 0, "chunk_M_tiles must be > 0");

    // Aux tensors: counts / global_expert_idx_table are small UINT32 vectors
    // the reader fetches via DRAM accessor. The reader does a single
    // noc_async_read_page(page=0, ...) and then indexes anywhere in
    // [0, num_global_experts), so the full vector must fit in one page. The
    // L1 scratch CB is sized to each tensor's aligned page (see the program
    // factory), while MAX_GLOBAL_EXPERTS caps the reader's supported index
    // space. This covers DeepSeek V3 (256), Kimi (384), and models up to the
    // cap. Validate the length here so larger expert counts produce a clean
    // assertion instead of silent OOB reads at runtime.
    for (const auto& [name, a] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"counts", t.counts}, {"global_expert_idx_table", t.global_expert_idx_table}}) {
        validate_index_tensor(a, name, t.x);
        const uint32_t num_entries = a.logical_shape()[-1];
        TT_FATAL(
            num_entries <= MAX_GLOBAL_EXPERTS,
            "{} length ({}) exceeds the maximum supported number of experts ({}) — "
            "the reader supports at most this many entries in its page-0 L1 scratch",
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
        validate_index_tensor(start, "expert_region_offsets", t.x);
        const auto& start_shape = start.logical_shape();
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
        TT_FATAL(out.storage_type() == tt::tt_metal::StorageType::DEVICE, "optional_output must be on device");
        TT_FATAL(out.buffer() != nullptr, "optional_output must have a buffer");
        TT_FATAL(out.device() == t.x.device(), "optional_output must be on the same device as x");
        TT_FATAL(out.layout() == tt::tt_metal::Layout::TILE, "optional_output must be TILE layout");
        TT_FATAL(is_dram_interleaved(out), "optional_output must be DRAM-interleaved");
        TT_FATAL(
            out.dtype() == t.x.dtype(), "optional_output dtype ({}) must match x dtype ({})", out.dtype(), t.x.dtype());
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
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(t.x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem));
}

UnifiedRoutedExpertFfnDeviceOperation::tensor_return_value_t
UnifiedRoutedExpertFfnDeviceOperation::create_output_tensors(const operation_attributes_t& op, const tensor_args_t& t) {
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
    bool stacked_packed_weights,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output,
    const std::optional<ttnn::Tensor>& expert_region_offsets) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::
        UnifiedRoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .chunk_M_tiles = chunk_M_tiles,
            .local_expert_id = local_expert_id,
            .stacked_packed_weights = stacked_packed_weights,
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .x = x,
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table,
            .optional_output = optional_output,
            .expert_region_offsets = expert_region_offsets});
}

}  // namespace ttnn::prim
