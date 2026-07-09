// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rotate_device_operation.hpp"
#include "fused_rotate_program_factory.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void FusedRotateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& x = inputs.x_flat;
    const auto& coef = inputs.coef_exp;
    TT_FATAL(
        x.storage_type() == StorageType::DEVICE && coef.storage_type() == StorageType::DEVICE,
        "fused_rotate operands must be on device");
    TT_FATAL(x.layout() == Layout::TILE && coef.layout() == Layout::TILE, "fused_rotate requires TILE layout");
    // bf16 or bf8_b; x and coef must share a format (the program factory derives one tile size
    // from x.dtype() for both CBs). bf8_b coef is parity-safe (orthogonal basis change, O(1) coefs)
    // and halves the [E,W] edge-activation DRAM traffic that dominates the bandwidth-bound replay.
    TT_FATAL(
        (x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::BFLOAT8_B) && coef.dtype() == x.dtype(),
        "fused_rotate requires bf16 or bf8_b inputs (x and coef same dtype)");

    const auto& xs = x.padded_shape();
    const auto& cs = coef.padded_shape();
    TT_FATAL(attrs.W % TILE_WIDTH == 0, "W ({}) must be a multiple of TILE_WIDTH", attrs.W);
    TT_FATAL(xs[-1] == attrs.n_in * attrs.W, "x_flat last dim {} != n_in*W {}", xs[-1], attrs.n_in * attrs.W);
    TT_FATAL(cs[-1] == attrs.nnz * TILE_WIDTH, "coef_exp last dim {} != nnz*32 {}", cs[-1], attrs.nnz * TILE_WIDTH);
    TT_FATAL(xs[-2] == cs[-2], "x_flat and coef_exp must have the same number of rows (edges)");
    TT_FATAL(attrs.deg.size() == attrs.n_out, "deg size {} != n_out {}", attrs.deg.size(), attrs.n_out);
    TT_FATAL(attrs.ks.size() == attrs.nnz && attrs.js.size() == attrs.nnz, "ks/js size must equal nnz");
    uint32_t sum = 0;
    for (auto d : attrs.deg) {
        sum += d;
    }
    TT_FATAL(sum == attrs.nnz, "sum(deg)={} != nnz={}", sum, attrs.nnz);
}

FusedRotateDeviceOperation::spec_return_value_t FusedRotateDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const auto& x = inputs.x_flat;
    ttnn::Shape out_shape(x.logical_shape());
    out_shape[-1] = attrs.n_out * attrs.W;
    return TensorSpec(out_shape, TensorLayout(x.dtype(), PageConfig(Layout::TILE), x.memory_config()));
}

FusedRotateDeviceOperation::tensor_return_value_t FusedRotateDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return create_device_tensor(compute_output_specs(attrs, inputs), inputs.x_flat.device());
}

ttsl::hash::hash_t FusedRotateDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    // The sparsity pattern (deg/ks/js) is set as RUNTIME args in create() but NOT refreshed in
    // override_runtime_arguments, so two calls that share shapes but differ in pattern (e.g. the
    // forward rotation grouped by output i vs. the backward g_in grouped by input j -- identical
    // [n_in,n_out,W,nnz] for a square rotation) MUST NOT share a cached program. Fold the pattern
    // into the hash so each distinct pattern gets its own program.
    uint64_t ph = 1469598103934665603ULL;  // FNV-1a
    auto mix = [&](uint32_t v) {
        ph = (ph ^ v) * 1099511628211ULL;
    };
    for (auto v : attrs.deg) {
        mix(v);
    }
    for (auto v : attrs.ks) {
        mix(v);
    }
    for (auto v : attrs.js) {
        mix(v);
    }
    return tt::tt_metal::operation::hash_operation<FusedRotateDeviceOperation>(
        attrs.n_in,
        attrs.n_out,
        attrs.W,
        attrs.nnz,
        static_cast<uint32_t>(ph),
        static_cast<uint32_t>(ph >> 32),
        inputs.x_flat.dtype(),
        inputs.x_flat.memory_config(),
        inputs.x_flat.padded_shape(),
        inputs.coef_exp.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_rotate(
    const Tensor& x_flat,
    const Tensor& coef_exp,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t W,
    const std::vector<uint32_t>& deg,
    const std::vector<uint32_t>& ks,
    const std::vector<uint32_t>& js) {
    using OperationType = ttnn::experimental::prim::FusedRotateDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .n_in = n_in,
        .n_out = n_out,
        .W = W,
        .nnz = static_cast<uint32_t>(ks.size()),
        .deg = deg,
        .ks = ks,
        .js = js};
    return ttnn::device_operation::launch<OperationType>(
        attrs, OperationType::tensor_args_t{.x_flat = x_flat, .coef_exp = coef_exp});
}

}  // namespace ttnn::prim
