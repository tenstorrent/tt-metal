// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lnbw_device_operation.hpp"
#include "lnbw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void LnBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& gy = inputs.gy;
    const auto& x = inputs.x;
    const auto& red = inputs.red;
    TT_FATAL(
        gy.storage_type() == StorageType::DEVICE && x.storage_type() == StorageType::DEVICE &&
            red.storage_type() == StorageType::DEVICE,
        "fused_ln_bw operands must be on device");
    TT_FATAL(
        gy.layout() == Layout::TILE && x.layout() == Layout::TILE && red.layout() == Layout::TILE,
        "fused_ln_bw requires TILE layout");
    TT_FATAL(
        gy.dtype() == DataType::BFLOAT16 && x.dtype() == DataType::BFLOAT16 && red.dtype() == DataType::BFLOAT16,
        "fused_ln_bw requires BFLOAT16 inputs");
    TT_FATAL(attrs.W % TILE_WIDTH == 0, "W ({}) must be a multiple of TILE_WIDTH", attrs.W);
    const auto& gs = gy.padded_shape();
    const auto& xsh = x.padded_shape();
    TT_FATAL(gs[-1] == attrs.W, "gy last dim {} != W {}", gs[-1], attrs.W);
    TT_FATAL(xsh[-1] == attrs.W, "x last dim {} != W {}", xsh[-1], attrs.W);
    TT_FATAL(gs[-2] == xsh[-2], "gy and x must have the same number of rows");
    const auto& n = inputs.n;
    const auto& gamma = inputs.gamma;
    TT_FATAL(
        n.storage_type() == StorageType::DEVICE && gamma.storage_type() == StorageType::DEVICE,
        "fused_ln_bw n/gamma must be on device");
    TT_FATAL(
        n.layout() == Layout::TILE && gamma.layout() == Layout::TILE, "fused_ln_bw n/gamma require TILE layout");
    TT_FATAL(
        n.dtype() == DataType::BFLOAT16 && gamma.dtype() == DataType::BFLOAT16,
        "fused_ln_bw n/gamma require BFLOAT16");
    TT_FATAL(n.padded_shape()[-1] == attrs.W, "n last dim {} != W {}", n.padded_shape()[-1], attrs.W);
    TT_FATAL(n.padded_shape()[-2] == gs[-2], "n and gy must have the same number of rows");
    TT_FATAL(gamma.padded_shape()[-1] == attrs.W, "gamma last dim {} != W {}", gamma.padded_shape()[-1], attrs.W);
}

LnBwDeviceOperation::spec_return_value_t LnBwDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    const auto& gy = inputs.gy;
    return TensorSpec(
        gy.logical_shape(), TensorLayout(gy.dtype(), PageConfig(Layout::TILE), gy.memory_config()));
}

LnBwDeviceOperation::tensor_return_value_t LnBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return create_device_tensor(compute_output_specs(attrs, inputs), inputs.gy.device());
}

ttsl::hash::hash_t LnBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return tt::tt_metal::operation::hash_operation<LnBwDeviceOperation>(
        attrs.W,
        attrs.eps_bits,
        inputs.gy.dtype(),
        inputs.gy.memory_config(),
        inputs.gy.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_ln_bw(
    const Tensor& gy,
    const Tensor& x,
    const Tensor& red,
    const Tensor& n,
    const Tensor& gamma,
    uint32_t W,
    uint32_t eps_bits) {
    using OperationType = ttnn::experimental::prim::LnBwDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{.W = W, .eps_bits = eps_bits};
    return ttnn::device_operation::launch<OperationType>(
        attrs, OperationType::tensor_args_t{.gy = gy, .x = x, .red = red, .n = n, .gamma = gamma});
}

}  // namespace ttnn::prim
