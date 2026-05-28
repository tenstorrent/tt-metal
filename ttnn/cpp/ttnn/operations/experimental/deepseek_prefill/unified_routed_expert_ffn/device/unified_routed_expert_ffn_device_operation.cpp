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
}  // namespace

void UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    TT_FATAL(t.x.storage_type() == tt::tt_metal::StorageType::DEVICE, "x must be on device");
    // x is restricted to BFLOAT8_B — the only dtype the existing callers
    // (TtRoutedExpert typecasts the dispatched buffer to BF8_B before this
    // op) and tests exercise. The kernel CB-size config can also accept
    // BFLOAT16, but that path is untested; reintroduce when a real caller
    // + PCC test for BF16 lands.
    TT_FATAL(t.x.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "x must be BFLOAT8_B, got {}", t.x.dtype());
    TT_FATAL(t.x.layout() == tt::tt_metal::Layout::TILE, "x must be TILE layout");
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

    // Weight tensors share x's storage / layout / memory contract — fail
    // host-side if the caller forgot to upload one, picked the wrong layout,
    // or sharded weights (the kernel reader assumes DRAM-interleaved).
    for (const auto& [name, w] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"gate_proj", t.gate_proj}, {"up_proj", t.up_proj}, {"down_proj", t.down_proj}}) {
        TT_FATAL(w.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
        TT_FATAL(is_dram_interleaved(w), "{} must be DRAM-interleaved", name);
    }

    // Aux tensors: counts / global_expert_idx_table are small UINT32 vectors
    // the reader fetches via DRAM accessor. The reader does a single
    // noc_async_read_page(page=0, ...) and then indexes anywhere in
    // [0, num_experts), so the full vector must fit in one page (= one tile
    // for TILE layout, 1024 uint32 entries). Validate here so larger expert
    // counts produce a clean assertion instead of silent OOB reads at runtime.
    for (const auto& [name, a] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"counts", t.counts}, {"global_expert_idx_table", t.global_expert_idx_table}}) {
        TT_FATAL(a.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(a.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32", name);
        TT_FATAL(is_dram_interleaved(a), "{} must be DRAM-interleaved", name);
        const uint32_t num_entries = a.logical_shape()[-1];
        TT_FATAL(
            num_entries <= tt::constants::TILE_HW,
            "{} length ({}) must fit in one tile ({} entries) — reader fetches "
            "only page 0 of this tensor",
            name,
            num_entries,
            tt::constants::TILE_HW);
    }
    TT_FATAL(
        op.local_expert_id < t.global_expert_idx_table.logical_shape()[-1],
        "local_expert_id ({}) >= idx_table size ({})",
        op.local_expert_id,
        t.global_expert_idx_table.logical_shape()[-1]);

    // optional_output: writer kernel computes per-core tile indices from
    // x.padded_shape[-2]/TILE; a smaller output buffer would overflow.
    if (t.optional_output.has_value()) {
        const auto& out = *t.optional_output;
        TT_FATAL(out.storage_type() == tt::tt_metal::StorageType::DEVICE, "optional_output must be on device");
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
        for (int i = 0; i < static_cast<int>(out_shape.rank()); ++i) {
            TT_FATAL(
                out_shape[i] == x_shape[i],
                "optional_output padded_shape[{}] ({}) must match x padded_shape[{}] ({})",
                i,
                out_shape[i],
                i,
                x_shape[i]);
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
        output_shape, tt::tt_metal::TensorLayout(t.x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem));
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
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .chunk_M_tiles = chunk_M_tiles,
            .local_expert_id = local_expert_id,
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .x = x,
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table,
            .optional_output = optional_output});
}

}  // namespace ttnn::prim
