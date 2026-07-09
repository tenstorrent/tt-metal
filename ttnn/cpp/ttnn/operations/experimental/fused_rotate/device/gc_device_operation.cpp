// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gc_device_operation.hpp"
#include "gc_program_factory.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void FusedGcDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& gout = inputs.gout;
    const auto& xin = inputs.xin;
    const auto& sel = inputs.sel;
    TT_FATAL(
        gout.storage_type() == StorageType::DEVICE && xin.storage_type() == StorageType::DEVICE &&
            sel.storage_type() == StorageType::DEVICE,
        "fused_rotate_gc operands must be on device");
    TT_FATAL(
        gout.layout() == Layout::TILE && xin.layout() == Layout::TILE && sel.layout() == Layout::TILE,
        "fused_rotate_gc requires TILE layout");
    // bf16 or bf8_b (one tile size for all CBs -> gout/xin/sel share a dtype).
    TT_FATAL(
        (gout.dtype() == DataType::BFLOAT16 || gout.dtype() == DataType::BFLOAT8_B) &&
            xin.dtype() == gout.dtype() && sel.dtype() == gout.dtype(),
        "fused_rotate_gc requires bf16 or bf8_b inputs (gout/xin/sel same dtype)");
    const auto& gs = gout.padded_shape();
    const auto& xs = xin.padded_shape();
    const auto& ss = sel.padded_shape();
    TT_FATAL(attrs.W % TILE_WIDTH == 0, "W ({}) must be a multiple of TILE_WIDTH", attrs.W);
    TT_FATAL(gs[-1] == attrs.n_out * attrs.W, "gout last dim {} != n_out*W {}", gs[-1], attrs.n_out * attrs.W);
    TT_FATAL(xs[-1] == attrs.n_in * attrs.W, "xin last dim {} != n_in*W {}", xs[-1], attrs.n_in * attrs.W);
    TT_FATAL(gs[-2] == xs[-2], "gout and xin must have the same number of rows (edges)");
    TT_FATAL(ss[-2] == TILE_HEIGHT && ss[-1] == 32 * TILE_WIDTH, "sel must be [32, 32*32]");
    TT_FATAL(attrs.is_.size() == attrs.nnz && attrs.js.size() == attrs.nnz, "is_/js size must equal nnz");
}

FusedGcDeviceOperation::spec_return_value_t FusedGcDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& gout = inputs.gout;
    ttnn::Shape out_shape(gout.logical_shape());
    out_shape[-1] = attrs.nnz;  // TILE layout pads to ceil(nnz/32)*32
    return TensorSpec(out_shape, TensorLayout(gout.dtype(), PageConfig(Layout::TILE), gout.memory_config()));
}

FusedGcDeviceOperation::tensor_return_value_t FusedGcDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return create_device_tensor(compute_output_specs(attrs, inputs), inputs.gout.device());
}

ttsl::hash::hash_t FusedGcDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    uint64_t ph = 1469598103934665603ULL;  // FNV-1a over the sparsity pattern (set as runtime args)
    auto mix = [&](uint32_t v) { ph = (ph ^ v) * 1099511628211ULL; };
    for (auto v : attrs.is_) {
        mix(v);
    }
    for (auto v : attrs.js) {
        mix(v);
    }
    return tt::tt_metal::operation::hash_operation<FusedGcDeviceOperation>(
        attrs.n_out,
        attrs.n_in,
        attrs.W,
        attrs.nnz,
        static_cast<uint32_t>(ph),
        static_cast<uint32_t>(ph >> 32),
        inputs.gout.dtype(),
        inputs.gout.memory_config(),
        inputs.gout.padded_shape(),
        inputs.xin.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_rotate_gc(
    const Tensor& gout,
    const Tensor& xin,
    const Tensor& sel,
    uint32_t n_out,
    uint32_t n_in,
    uint32_t W,
    const std::vector<uint32_t>& is_,
    const std::vector<uint32_t>& js) {
    using OperationType = ttnn::experimental::prim::FusedGcDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .n_out = n_out,
        .n_in = n_in,
        .W = W,
        .nnz = static_cast<uint32_t>(is_.size()),
        .is_ = is_,
        .js = js};
    return ttnn::device_operation::launch<OperationType>(
        attrs, OperationType::tensor_args_t{.gout = gout, .xin = xin, .sel = sel});
}

}  // namespace ttnn::prim
