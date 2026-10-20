// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_mix_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
void check_tensor(const Tensor& t, const char* name) {
    TT_FATAL(t.dtype() == tt::tt_metal::DataType::FLOAT32, "mhc_post_mix: {} must be FLOAT32", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "mhc_post_mix: {} must be TILE layout", name);
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "mhc_post_mix: {} must be on device", name);
    TT_FATAL(
        t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "mhc_post_mix: {} must be DRAM/L1 interleaved",
        name);
}
}  // namespace

void MhcPostMixDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    check_tensor(t.x, "x");
    check_tensor(t.y, "y");
    check_tensor(t.post, "post");
    check_tensor(t.comb, "comb");
    const uint32_t n = args.n;
    TT_FATAL(
        n >= 1 && n * n + n <= tt::constants::TILE_WIDTH,
        "mhc_post_mix: n*n + n ({}) must fit one tile row",
        n * n + n);
    const auto& xs = t.x.padded_shape();
    const auto& ys = t.y.padded_shape();
    TT_FATAL(xs.rank() == 4 && ys.rank() == 4, "mhc_post_mix: x and y must be rank 4");
    TT_FATAL(xs[0] == 1 && xs[1] == 1 && ys[0] == 1 && ys[1] == 1, "mhc_post_mix: x and y leading dims must be 1");
    const uint32_t T = xs[2];
    const uint32_t C = ys[3];
    TT_FATAL(
        T % tt::constants::TILE_HEIGHT == 0 && T == ys[2],
        "mhc_post_mix: x/y rows ({}, {}) must match and be tile aligned",
        T,
        ys[2]);
    TT_FATAL(
        C % tt::constants::TILE_WIDTH == 0 && xs[3] == n * C,
        "mhc_post_mix: x width ({}) must be n*C with C = {}",
        xs[3],
        C);
    TT_FATAL(
        t.post.padded_shape()[-2] == T && t.comb.padded_shape()[-2] == T,
        "mhc_post_mix: post/comb rows must match x rows ({})",
        T);
    TT_FATAL(
        t.post.padded_shape()[-1] == tt::constants::TILE_WIDTH &&
            t.comb.padded_shape()[-1] == tt::constants::TILE_WIDTH,
        "mhc_post_mix: post/comb must be one tile wide");
    TT_FATAL(
        t.post.logical_shape()[-1] >= n && t.comb.logical_shape()[-1] >= n * n,
        "mhc_post_mix: post needs {} columns, comb {}",
        n,
        n * n);
    TT_FATAL(
        t.x.device() == t.y.device() && t.x.device() == t.post.device() && t.x.device() == t.comb.device(),
        "mhc_post_mix: all inputs must be on the same device");
}

MhcPostMixDeviceOperation::spec_return_value_t MhcPostMixDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const auto& xl = t.x.logical_shape();
    const uint32_t C = t.y.padded_shape()[3];
    ttnn::Shape shape = args.collapse ? ttnn::Shape({xl[0], xl[1], xl[2], C}) : xl;
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcPostMixDeviceOperation::tensor_return_value_t MhcPostMixDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.x.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_post_mix(
    const Tensor& x, const Tensor& y, const Tensor& post, const Tensor& comb, uint32_t n, bool collapse) {
    using OperationType = ttnn::experimental::prim::MhcPostMixDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{n, collapse};
    auto tensor_args = OperationType::tensor_args_t{x, y, post, comb};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
