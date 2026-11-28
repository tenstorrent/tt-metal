// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_residual_device_operation.hpp"
#include "rms_norm_residual_table.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/program_cache_l1.hpp"

namespace ttnn::experimental::llama31_8b {

namespace {
bool is_dram_interleaved(const Tensor& t) {
    return t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           t.memory_config().buffer_type() == BufferType::DRAM;
}

std::uint32_t tile_bytes(DataType dtype) {
    return tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(dtype));
}

const rms_norm_residual_table::Entry* table_entry(const Tensor& input) {
    const auto grid = input.device()->compute_with_storage_grid_size();
    return rms_norm_residual_table::find(grid.x, grid.y, input.padded_shape()[-2], input.padded_shape()[-1]);
}

std::uint64_t buffer_bytes(const rms_norm_residual_table::Entry& entry, const Tensor& input, const Tensor& residual) {
    return rms_norm_residual_table::cb_bytes(
        entry,
        {tile_bytes(input.dtype()),
         tile_bytes(residual.dtype()),
         tile_bytes(DataType::BFLOAT16),
         tile_bytes(DataType::BFLOAT16),
         tile_bytes(DataType::FLOAT32),
         tile_bytes(DataType::BFLOAT16)});
}
}  // namespace

bool rms_norm_residual_fits(const Tensor& input, const Tensor& residual) {
    if (input.storage_type() != StorageType::DEVICE || input.device()->arch() != tt::ARCH::BLACKHOLE ||
        input.padded_shape().rank() != 4) {
        return false;
    }
    const auto* entry = table_entry(input);
    return entry != nullptr && buffer_bytes(*entry, input, residual) <=
                                   ttnn::operations::core::available_program_l1_capacity(input.device());
}

void RmsNormResidualOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& residual = tensor_args.residual;
    const auto& weight = tensor_args.weight;
    const auto& residual_out = tensor_args.residual_out;
    TT_FATAL(input.device()->arch() == tt::ARCH::BLACKHOLE, "llama31_8b.rms_norm_residual: Blackhole only");
    for (const Tensor* t : {&input, &residual, &weight, &residual_out}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "llama31_8b.rms_norm_residual: tensors must be on device");
        TT_FATAL(t->is_allocated(), "llama31_8b.rms_norm_residual: tensors must be allocated");
    }
    const auto& shape = input.padded_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1,
        "llama31_8b.rms_norm_residual: input must be [1, 1, S, W]");
    TT_FATAL(
        shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % (32 * tt::constants::TILE_WIDTH) == 0,
        "llama31_8b.rms_norm_residual: S must be whole tiles and W a multiple of 1024");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 && input.layout() == Layout::TILE && is_dram_interleaved(input),
        "llama31_8b.rms_norm_residual: input must be bfloat16 TILE, DRAM interleaved");
    TT_FATAL(
        residual.padded_shape() == shape && residual.layout() == Layout::TILE && is_dram_interleaved(residual) &&
            (residual.dtype() == DataType::BFLOAT16 || residual.dtype() == DataType::BFLOAT8_B),
        "llama31_8b.rms_norm_residual: residual must match input, bfloat16 or bfloat8_b TILE, DRAM interleaved");
    TT_FATAL(
        residual_out.padded_shape() == shape && residual_out.dtype() == DataType::BFLOAT16 &&
            residual_out.layout() == Layout::TILE && is_dram_interleaved(residual_out),
        "llama31_8b.rms_norm_residual: residual_out must match input, bfloat16 TILE, DRAM interleaved");
    TT_FATAL(
        weight.dtype() == DataType::BFLOAT16 && weight.layout() == Layout::ROW_MAJOR &&
            weight.padded_shape()[-1] == tt::constants::TILE_WIDTH && weight.physical_volume() == shape[-1],
        "llama31_8b.rms_norm_residual: weight must be bfloat16 ROW_MAJOR [1, 1, W / 32, 32]");
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "llama31_8b.rms_norm_residual: fp32_dest_acc_en must be set");
    const auto* entry = table_entry(input);
    TT_FATAL(entry != nullptr, "llama31_8b.rms_norm_residual: no entry for this shape and grid");
    TT_FATAL(
        buffer_bytes(*entry, input, residual) <= ttnn::operations::core::available_program_l1_capacity(input.device()),
        "llama31_8b.rms_norm_residual: not enough free L1 for this program; use ttnn.add + ttnn.rms_norm");
}

void RmsNormResidualOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    for (const Tensor* t :
         {&tensor_args.input, &tensor_args.residual, &tensor_args.weight, &tensor_args.residual_out}) {
        TT_FATAL(t->is_allocated(), "llama31_8b.rms_norm_residual: tensors must be allocated");
    }
}

RmsNormResidualOperation::spec_return_value_t RmsNormResidualOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.residual_out.tensor_spec();
}

RmsNormResidualOperation::tensor_return_value_t RmsNormResidualOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_rms_norm_residual(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& weight,
    const Tensor& residual_out,
    float eps,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::llama31_8b::RmsNormResidualOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{eps, compute_kernel_config},
        OperationType::tensor_args_t{input, residual, weight, residual_out});
}
}  // namespace ttnn::prim
