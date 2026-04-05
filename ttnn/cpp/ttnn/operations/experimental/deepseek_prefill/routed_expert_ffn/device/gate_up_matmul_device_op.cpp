// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_up_matmul_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

void GateUpMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& inputs) {
    TT_FATAL(inputs.x.layout() == tt::tt_metal::Layout::TILE, "x must be TILE layout");
    TT_FATAL(inputs.gate_proj.layout() == tt::tt_metal::Layout::TILE, "gate_proj must be TILE layout");
    TT_FATAL(inputs.up_proj.layout() == tt::tt_metal::Layout::TILE, "up_proj must be TILE layout");
    TT_FATAL(
        inputs.gate_proj.logical_shape() == inputs.up_proj.logical_shape(),
        "gate_proj and up_proj must have identical shapes");
    TT_FATAL(
        inputs.x.logical_shape()[-1] == inputs.gate_proj.logical_shape()[-2],
        "x K dim ({}) must match gate_proj K dim ({})",
        inputs.x.logical_shape()[-1],
        inputs.gate_proj.logical_shape()[-2]);
}

void GateUpMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*inputs*/) {}

GateUpMatmulDeviceOperation::spec_return_value_t GateUpMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& x_shape = inputs.x.logical_shape();
    const auto& w_shape = inputs.gate_proj.logical_shape();

    // Output shape: same leading dims as x but last dim replaced by N (weight output dim)
    ttnn::SmallVector<uint32_t> out_dims;
    for (size_t i = 0; i + 1 < x_shape.rank(); ++i) {
        out_dims.push_back(x_shape[i]);
    }
    out_dims.push_back(w_shape[-1]);
    auto out_shape = ttnn::Shape(out_dims);

    auto spec = TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), attrs.output_mem_config));
    return {spec, spec};
}

GateUpMatmulDeviceOperation::topology_return_value_t GateUpMatmulDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& inputs) {
    const auto& topology = inputs.x.tensor_topology();
    return {topology, topology};
}

GateUpMatmulDeviceOperation::tensor_return_value_t GateUpMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto specs = compute_output_specs(attrs, inputs);
    return {create_device_tensor(specs[0], inputs.x.device()), create_device_tensor(specs[1], inputs.x.device())};
}

tt::stl::hash::hash_t GateUpMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return tt::tt_metal::operation::hash_operation<GateUpMatmulDeviceOperation>(
        attrs, inputs.x.dtype(), inputs.gate_proj.dtype(), inputs.x.memory_config(), inputs.x.padded_shape());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn::prim {

std::array<ttnn::Tensor, 2> gate_up_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& /*compute_kernel_config*/) {
    using Op = ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::GateUpMatmulDeviceOperation;

    const auto& x_shape = x.padded_shape();
    const auto& w_shape = gate_proj.padded_shape();

    // M — number of tokens (sequence length)
    // K — shared inner dim: x's last dim == gate_proj's second-to-last dim (hidden size)
    // N — output columns: gate_proj's last dim (intermediate/expert hidden size)
    uint32_t Mt = (x.physical_volume() / x_shape[-1]) / tt::constants::TILE_HEIGHT;
    uint32_t Kt = x_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Nt = w_shape[-1] / tt::constants::TILE_WIDTH;

    // Tune these parameters for best performance
    constexpr uint32_t K_block_size = 4;
    constexpr uint32_t N_block_size = 4;
    constexpr uint32_t M_block_size = 1;
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t subblock_w = 4;

    Op::operation_attributes_t attrs{
        .Mt = Mt,
        .Kt = Kt,
        .Nt = Nt,
        .M_block_size = M_block_size,
        .K_block_size = K_block_size,
        .N_block_size = N_block_size,
        .subblock_h = subblock_h,
        .subblock_w = subblock_w,
        .output_mem_config = x.memory_config(),
        .output_dtype = x.dtype()};

    Op::tensor_args_t tensor_args{x, gate_proj, up_proj};

    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
