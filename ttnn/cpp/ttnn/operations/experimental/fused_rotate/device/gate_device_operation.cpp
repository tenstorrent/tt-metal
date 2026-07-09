// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_device_operation.hpp"
#include "gate_program_factory.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void GateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& a = inputs.a;
    const auto& gate = inputs.gate;
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && gate.storage_type() == StorageType::DEVICE,
        "fused_gate operands must be on device");
    TT_FATAL(a.layout() == Layout::TILE && gate.layout() == Layout::TILE, "fused_gate requires TILE layout");
    // bf16 or bf8_b (one tile size for all CBs -> a and gate share a dtype). bf8 halves the
    // [E,W] edge-activation traffic on the bandwidth-bound replay.
    TT_FATAL(
        (a.dtype() == DataType::BFLOAT16 || a.dtype() == DataType::BFLOAT8_B) && gate.dtype() == a.dtype(),
        "fused_gate requires bf16 or bf8_b (a and gate same dtype)");
    TT_FATAL(attrs.Wt == attrs.Ht + attrs.Gt, "Wt ({}) must equal Ht ({}) + Gt ({})", attrs.Wt, attrs.Ht, attrs.Gt);
    const auto& as_ = a.padded_shape();
    const auto& gs = gate.padded_shape();
    TT_FATAL(as_[-1] == attrs.Wt * TILE_WIDTH, "a last dim {} != Wt*32 {}", as_[-1], attrs.Wt * TILE_WIDTH);
    TT_FATAL(gs[-1] == attrs.Gt * TILE_WIDTH, "gate last dim {} != Gt*32 {}", gs[-1], attrs.Gt * TILE_WIDTH);
    TT_FATAL(as_[-2] == gs[-2], "a and gate must have the same number of rows");
    if (attrs.mode == 1) {
        const auto& b = inputs.b;
        TT_FATAL(b.storage_type() == StorageType::DEVICE && b.layout() == Layout::TILE, "fused_gate b device/TILE");
        TT_FATAL(b.dtype() == a.dtype(), "fused_gate b must match a dtype");
        TT_FATAL(b.padded_shape()[-2] == as_[-2], "b and a must have the same number of rows");
    }
}

GateDeviceOperation::spec_return_value_t GateDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    const auto& a = inputs.a;
    return TensorSpec(a.logical_shape(), TensorLayout(a.dtype(), PageConfig(Layout::TILE), a.memory_config()));
}

GateDeviceOperation::tensor_return_value_t GateDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return create_device_tensor(compute_output_specs(attrs, inputs), inputs.a.device());
}

ttsl::hash::hash_t GateDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return tt::tt_metal::operation::hash_operation<GateDeviceOperation>(
        attrs.Wt, attrs.Gt, attrs.Ht, attrs.mode, inputs.a.dtype(), inputs.a.memory_config(), inputs.a.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_gate(
    const Tensor& a, const Tensor& gate, const Tensor& b, uint32_t Wt, uint32_t Gt, uint32_t Ht, uint32_t mode) {
    using OperationType = ttnn::experimental::prim::GateDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{.Wt = Wt, .Gt = Gt, .Ht = Ht, .mode = mode};
    return ttnn::device_operation::launch<OperationType>(
        attrs, OperationType::tensor_args_t{.a = a, .gate = gate, .b = b});
}

}  // namespace ttnn::prim
