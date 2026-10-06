// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "practice_routed_expert_device_operation.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

namespace {

bool is_dram_interleaved(const ttnn::Tensor& tensor) {
    const auto& mem_cfg = tensor.memory_config();
    return mem_cfg.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_cfg.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_operand(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout, got {}", name, tensor.layout());
    TT_FATAL(is_dram_interleaved(tensor), "{} must be DRAM interleaved", name);

    const auto& shape = tensor.logical_shape();
    TT_FATAL(shape.rank() == 2, "{} must be 2D, got shape {}", name, shape);
    // The kernels walk whole tiles and never mask a partial one.
    TT_FATAL(
        shape[0] % tt::constants::TILE_HEIGHT == 0 && shape[1] % tt::constants::TILE_WIDTH == 0,
        "{} dims must be multiples of the tile size, got shape {}",
        name,
        shape);
}

}  // namespace

void PracticeRoutedExpertDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& [x, w_gate, w_up, w_down] = tensor_args;

    validate_operand(x, "x");
    validate_operand(w_gate, "w_gate");
    validate_operand(w_up, "w_up");
    validate_operand(w_down, "w_down");
    TT_FATAL(
        x.device()->arch() == tt::ARCH::BLACKHOLE,
        "practice_routed_expert needs Blackhole: SiTU-GLU's SFPU op exists there only");

    TT_FATAL(
        x.dtype() == tt::tt_metal::DataType::BFLOAT16 || x.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "x must be BFLOAT16 or BFLOAT8_B, got {}",
        x.dtype());
    TT_FATAL(
        w_gate.dtype() == tt::tt_metal::DataType::BFLOAT16 || w_gate.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
            w_gate.dtype() == tt::tt_metal::DataType::BFLOAT4_B,
        "weights must be BFLOAT16, BFLOAT8_B or BFLOAT4_B, got {}",
        w_gate.dtype());
    // The compute kernel programs one weight format for the gate and up matmuls together.
    TT_FATAL(
        w_up.dtype() == w_gate.dtype() && w_down.dtype() == w_gate.dtype(),
        "w_gate, w_up and w_down must share one dtype, got {}, {} and {}",
        w_gate.dtype(),
        w_up.dtype(),
        w_down.dtype());

    const auto& x_shape = x.logical_shape();
    const auto& gate_shape = w_gate.logical_shape();
    const uint32_t emb_dim = x_shape[1];
    const uint32_t hidden_dim = gate_shape[1];
    TT_FATAL(gate_shape[0] == emb_dim, "w_gate must be (K, N) with K = x's width {}, got {}", emb_dim, gate_shape);
    TT_FATAL(
        w_up.logical_shape() == gate_shape,
        "w_up must match w_gate's shape {}, got {}",
        gate_shape,
        w_up.logical_shape());
    TT_FATAL(
        w_down.logical_shape()[0] == hidden_dim && w_down.logical_shape()[1] == emb_dim,
        "w_down must be (N, K) = ({}, {}), got {}",
        hidden_dim,
        emb_dim,
        w_down.logical_shape());
}

void PracticeRoutedExpertDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

PracticeRoutedExpertDeviceOperation::spec_return_value_t PracticeRoutedExpertDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x;
    const auto mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        x.logical_shape(),
        tt::tt_metal::TensorLayout(x.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
}

PracticeRoutedExpertDeviceOperation::tensor_return_value_t PracticeRoutedExpertDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.x.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert

namespace ttnn::prim {

ttnn::Tensor practice_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::practice_routed_expert::PracticeRoutedExpertDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{.x = x, .w_gate = w_gate, .w_up = w_up, .w_down = w_down});
}

}  // namespace ttnn::prim
