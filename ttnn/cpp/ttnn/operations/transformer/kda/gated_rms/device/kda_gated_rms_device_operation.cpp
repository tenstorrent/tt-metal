// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_gated_rms_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

void check_tiled_device(const Tensor& tensor, const char* name, DataType dtype) {
    TT_FATAL(tensor.layout() == Layout::TILE, "KDA gated RMS: {} must be TILE layout", name);
    TT_FATAL(tensor.dtype() == dtype, "KDA gated RMS: {} has wrong dtype", name);
    TT_FATAL(tensor.buffer() != nullptr, "KDA gated RMS: {} must be on device", name);
}

void check_intermediate(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.layout() == Layout::TILE, "KDA gated RMS: {} must be TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "KDA gated RMS: {} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(tensor.buffer() != nullptr, "KDA gated RMS: {} must be on device", name);
}

}  // namespace

KdaGatedRmsOperation::program_factory_t KdaGatedRmsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaGatedRmsProgramFactory{};
}

void KdaGatedRmsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_intermediate(in.input, "input");
    check_tiled_device(in.gate, "gate", DataType::BFLOAT16);
    check_tiled_device(in.weight, "weight", DataType::BFLOAT16);
    TT_FATAL(
        attrs.output_dtype == DataType::FLOAT32 || attrs.output_dtype == DataType::BFLOAT16,
        "KDA gated RMS output_dtype must be FLOAT32 or BFLOAT16");
    const auto& input_shape = in.input.logical_shape();
    const auto& gate_shape = in.gate.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "KDA gated RMS input must be [B*H,T,V]");
    TT_FATAL(gate_shape.rank() == 3, "KDA gated RMS gate must be [B,T,H*V]");
    TT_FATAL(
        input_shape[0] == attrs.batch * attrs.num_heads && input_shape[1] == attrs.sequence &&
            input_shape[2] == attrs.value_dim,
        "KDA gated RMS input shape does not match attributes");
    TT_FATAL(
        gate_shape[0] == attrs.batch && gate_shape[1] == attrs.sequence &&
            gate_shape[2] == attrs.num_heads * attrs.value_dim,
        "KDA gated RMS gate shape does not match attributes");
    TT_FATAL(in.weight.logical_volume() == attrs.value_dim, "KDA gated RMS weight volume must equal V");
    TT_FATAL(attrs.sequence % tt::constants::TILE_HEIGHT == 0, "KDA gated RMS sequence must be tile aligned");
    TT_FATAL(attrs.value_dim % tt::constants::TILE_WIDTH == 0, "KDA gated RMS value_dim must be tile aligned");
}

KdaGatedRmsOperation::spec_return_value_t KdaGatedRmsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    return {TensorSpec(
        Shape({attrs.batch, attrs.sequence, attrs.num_heads * attrs.value_dim}),
        TensorLayout(attrs.output_dtype, PageConfig(Layout::TILE), attrs.output_mem_config))};
}

KdaGatedRmsOperation::tensor_return_value_t KdaGatedRmsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.input.device())};
}

Tensor kda_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "KDA gated RMS input must be [B*H,T,V]");
    TT_FATAL(num_heads > 0, "KDA gated RMS num_heads must be positive");
    TT_FATAL(input_shape[0] % num_heads == 0, "KDA gated RMS leading dimension must be divisible by num_heads");
    const uint32_t batch = input_shape[0] / num_heads;
    auto results = ttnn::device_operation::launch<KdaGatedRmsOperation>(
        KdaGatedRmsParams{
            .batch = batch,
            .num_heads = num_heads,
            .sequence = static_cast<uint32_t>(input_shape[1]),
            .value_dim = static_cast<uint32_t>(input_shape[2]),
            .epsilon = epsilon,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config},
        KdaGatedRmsInputs{.input = input, .gate = gate, .weight = weight});
    return results[0];
}

}  // namespace ttnn::prim
