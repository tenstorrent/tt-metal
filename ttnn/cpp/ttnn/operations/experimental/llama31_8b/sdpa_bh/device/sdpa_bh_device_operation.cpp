// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/sdpa_bh/device/sdpa_bh_device_operation.hpp"

#include <array>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

namespace {

struct ShapeRow {
    uint32_t seq, q_chunk, k_chunk;
};

constexpr std::array<ShapeRow, 1> kShapes{{{1024, 64, 64}}};

bool dram_interleaved_bf8_tile(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.dtype() == DataType::BFLOAT8_B && t.layout() == Layout::TILE &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           t.memory_config().buffer_type() == BufferType::DRAM;
}

}  // namespace

bool sdpa_bh_supported(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    if (input_tensor_q.storage_type() != StorageType::DEVICE ||
        input_tensor_q.device()->arch() != tt::ARCH::BLACKHOLE || !program_config.has_value() ||
        program_config->exp_approx_mode.value_or(false) || program_config->sub_core_grids.has_value()) {
        return false;
    }
    for (const Tensor* t : {&input_tensor_q, &input_tensor_k, &input_tensor_v}) {
        if (!dram_interleaved_bf8_tile(*t)) {
            return false;
        }
    }
    const auto q = input_tensor_q.padded_shape();
    const auto k = input_tensor_k.padded_shape();
    const auto v = input_tensor_v.padded_shape();
    if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || q[0] != 1 || k[0] != 1 || v[0] != 1 || q[1] != 32 ||
        k[1] != 8 || v[1] != 8 || q[3] != 128 || k[3] != 128 || v[3] != 128 || q[2] != k[2] || k[2] != v[2] ||
        input_tensor_q.logical_shape()[2] != q[2] || input_tensor_k.logical_shape()[2] != k[2]) {
        return false;
    }
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor_q.device()->arch(), compute_kernel_config);
    if (math_fidelity != tt::tt_metal::MathFidelity::HiFi4 || math_approx_mode || !fp32_dest_acc_en ||
        dst_full_sync_en) {
        return false;
    }
    const CoreCoord grid = program_config->compute_with_storage_grid_size;
    if (grid.x != 11 || grid.y != 10) {
        return false;
    }
    for (const auto& row : kShapes) {
        if (row.seq == q[2] && row.q_chunk == program_config->q_chunk_size &&
            row.k_chunk == program_config->k_chunk_size) {
            return true;
        }
    }
    return false;
}

void SdpaBhOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        operation_attributes.is_causal && tensor_args.v.has_value() &&
            sdpa_bh_supported(
                tensor_args.q,
                tensor_args.k,
                tensor_args.v.value(),
                operation_attributes.program_config,
                operation_attributes.compute_kernel_config),
        "llama31_8b.sdpa_bh: only the Llama-3.1-8B Blackhole prefill shapes are supported; check "
        "ttnn.experimental.llama31_8b.sdpa_bh_supported first and use ttnn.transformer.scaled_dot_product_attention "
        "otherwise");
    ttnn::prim::SDPAOperation::validate_on_program_cache_miss(operation_attributes, tensor_args);
}

SdpaBhOperation::spec_return_value_t SdpaBhOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttnn::prim::SDPAOperation::compute_output_specs(operation_attributes, tensor_args);
}

SdpaBhOperation::tensor_return_value_t SdpaBhOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_sdpa_bh(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool output_concat_heads) {
    using OperationType = ttnn::experimental::llama31_8b::SdpaBhOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = program_config,
            .is_causal = true,
            .compute_kernel_config = compute_kernel_config,
            .output_concat_heads = output_concat_heads,
        },
        OperationType::tensor_args_t{.q = input_tensor_q, .k = input_tensor_k, .v = input_tensor_v});
}
}  // namespace ttnn::prim
