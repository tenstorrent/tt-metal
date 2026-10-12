// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn_device_operation.hpp"

#include <initializer_list>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

namespace {
namespace unified = unified_routed_expert_ffn;

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

bool is_dram_interleaved(const ttnn::Tensor& t) {
    const auto& mem = t.memory_config();
    return mem.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

// One expert owning all of x's rows: the program is sized for M rows, and the three layout
// vectors say the expert holds M tokens from row 0.
unified::UnifiedRoutedExpertFfnParams to_unified_params(
    const SharedExpertFfnParams& op, const SharedExpertFfnInputs& t) {
    return unified::UnifiedRoutedExpertFfnParams{
        .m_tiles = static_cast<uint32_t>(t.x.padded_shape()[-2]) / TILE,
        .experts_per_chip = 1,
        .x_is_row_major = t.x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        .activation = op.activation,
        .fuse_bias = false,
        .compute_kernel_config = op.compute_kernel_config,
        .core_grid_origin = op.core_grid_origin,
    };
}

unified::UnifiedRoutedExpertFfnInputs to_unified_inputs(const SharedExpertFfnInputs& t, const Tensor& output) {
    return unified::UnifiedRoutedExpertFfnInputs{
        .x = t.x,
        .gate_projs = {t.gate_proj},
        .up_projs = {t.up_proj},
        .down_projs = {t.down_proj},
        .counts = t.expert_token_counts,
        .global_expert_idx_table = t.global_expert_idx_table,
        .output = output,
        .expert_region_offsets = t.expert_region_offsets,
    };
}
}  // namespace

tt::tt_metal::ProgramDescriptor SharedExpertFfnProgramFactory::create_descriptor(
    const SharedExpertFfnParams& op, const SharedExpertFfnInputs& t, Tensor& tensor_return_value) {
    return unified::UnifiedRoutedExpertFfnProgramFactory::create_descriptor(
        to_unified_params(op, t), to_unified_inputs(t, tensor_return_value), tensor_return_value);
}

void SharedExpertFfnProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SharedExpertFfnParams& op,
    const SharedExpertFfnInputs& t,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    unified::UnifiedRoutedExpertFfnProgramFactory::override_runtime_arguments(
        program, to_unified_params(op, t), to_unified_inputs(t, tensor_return_value), tensor_return_value, coord);
}

void SharedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& t) {
    const auto& x = t.x;
    TT_FATAL(x.storage_type() == ttnn::StorageType::DEVICE, "shared_expert_ffn: x must be on device");
    TT_FATAL(
        x.device()->arch() == tt::ARCH::BLACKHOLE,
        "shared_expert_ffn: implemented for Blackhole only, got arch {}",
        x.device()->arch());
    const bool x_tile_ok = x.layout() == tt::tt_metal::Layout::TILE && (x.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
                                                                        x.dtype() == tt::tt_metal::DataType::BFLOAT8_B);
    const bool x_rm_ok = x.layout() == tt::tt_metal::Layout::ROW_MAJOR && x.dtype() == tt::tt_metal::DataType::BFLOAT16;
    TT_FATAL(
        x_tile_ok || x_rm_ok,
        "shared_expert_ffn: x must be TILE BFLOAT16 / BFLOAT8_B or ROW_MAJOR BFLOAT16, got {} {}",
        x.layout(),
        x.dtype());
    TT_FATAL(is_dram_interleaved(x), "shared_expert_ffn: x must be DRAM-interleaved");
    const auto& x_shape = x.padded_shape();
    TT_FATAL(x_shape.rank() >= 2, "shared_expert_ffn: x must have rank >= 2, got {}", x_shape.rank());
    for (int i = 0; i < static_cast<int>(x_shape.rank()) - 2; ++i) {
        TT_FATAL(x.logical_shape()[i] == 1, "shared_expert_ffn: x leading dim {} must be 1, got {}", i, x_shape[i]);
    }
    TT_FATAL(
        x_shape[-2] > 0 && x_shape[-2] % TILE == 0,
        "shared_expert_ffn: x rows ({}) must be a positive multiple of {}",
        x_shape[-2],
        TILE);

    for (const auto& [name, w] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"gate_proj", t.gate_proj}, {"up_proj", t.up_proj}, {"down_proj", t.down_proj}}) {
        TT_FATAL(w.storage_type() == ttnn::StorageType::DEVICE, "shared_expert_ffn: {} must be on device", name);
        TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "shared_expert_ffn: {} must be TILE layout", name);
        TT_FATAL(is_dram_interleaved(w), "shared_expert_ffn: {} must be DRAM-interleaved", name);
    }
    const auto& gate_shape = t.gate_proj.padded_shape();
    const auto& up_shape = t.up_proj.padded_shape();
    const auto& down_shape = t.down_proj.padded_shape();
    TT_FATAL(
        gate_shape[-2] == x_shape[-1] && up_shape[-2] == x_shape[-1],
        "shared_expert_ffn: gate/up K ({}, {}) must equal x's emb ({})",
        gate_shape[-2],
        up_shape[-2],
        x_shape[-1]);
    TT_FATAL(
        gate_shape[-1] == up_shape[-1] && down_shape[-2] == gate_shape[-1],
        "shared_expert_ffn: gate N ({}), up N ({}) and down K ({}) must all be the hidden dim",
        gate_shape[-1],
        up_shape[-1],
        down_shape[-2]);
    TT_FATAL(
        t.down_proj.logical_shape()[-1] % TILE == 0 && down_shape[-1] <= x_shape[-1],
        "shared_expert_ffn: down N ({}) must be tile-aligned and at most x's emb ({})",
        t.down_proj.logical_shape()[-1],
        x_shape[-1]);

    // The unified reader / writer fetch page 0 of each vector and index entry 0 (the one expert).
    for (const auto& [name, v] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"expert_token_counts", t.expert_token_counts},
             {"global_expert_idx_table", t.global_expert_idx_table},
             {"expert_region_offsets", t.expert_region_offsets}}) {
        TT_FATAL(v.storage_type() == ttnn::StorageType::DEVICE, "shared_expert_ffn: {} must be on device", name);
        TT_FATAL(v.dtype() == tt::tt_metal::DataType::UINT32, "shared_expert_ffn: {} must be UINT32", name);
        TT_FATAL(v.layout() == tt::tt_metal::Layout::ROW_MAJOR, "shared_expert_ffn: {} must be ROW_MAJOR", name);
        TT_FATAL(is_dram_interleaved(v), "shared_expert_ffn: {} must be DRAM-interleaved", name);
        TT_FATAL(
            v.logical_shape()[-1] >= 1 && v.logical_shape()[-1] <= unified::MAX_GLOBAL_EXPERTS,
            "shared_expert_ffn: {} must hold 1..{} entries, got {}",
            name,
            unified::MAX_GLOBAL_EXPERTS,
            v.logical_shape()[-1]);
    }
    TT_FATAL(
        t.expert_region_offsets.logical_shape()[-1] == t.expert_token_counts.logical_shape()[-1],
        "shared_expert_ffn: expert_region_offsets ({}) and expert_token_counts ({}) must be the same length",
        t.expert_region_offsets.logical_shape()[-1],
        t.expert_token_counts.logical_shape()[-1]);
}

void SharedExpertFfnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

SharedExpertFfnDeviceOperation::spec_return_value_t SharedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    // Always a fresh bf8_b TILE tensor, never x: x is typically read by other ops concurrently
    // (the MoE dispatch), and down N may be narrower than emb.
    ttnn::Shape shape = t.x.logical_shape();
    shape[-1] = t.down_proj.logical_shape()[-1];
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT8_B,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

SharedExpertFfnDeviceOperation::tensor_return_value_t SharedExpertFfnDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(op, t), t.x.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

namespace ttnn::prim {

ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn::RoutedExpertActivation activation,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const CoreCoord& core_grid_origin) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn::SharedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .activation = activation,
            .compute_kernel_config = compute_kernel_config,
            .core_grid_origin = core_grid_origin},
        OperationType::tensor_args_t{
            .x = x,
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .expert_token_counts = expert_token_counts,
            .global_expert_idx_table = global_expert_idx_table,
            .expert_region_offsets = expert_region_offsets});
}

}  // namespace ttnn::prim
