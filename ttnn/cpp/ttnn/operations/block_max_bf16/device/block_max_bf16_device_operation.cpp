// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_max_bf16_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::block_max_bf16 {

using namespace tt::tt_metal;

namespace {

// Python _dram_rm.
void check_dram_rm(const Tensor& t, const char* what, DataType dtype) {
    const auto& mc = t.memory_config();
    TT_FATAL(
        mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "block_select: {} must be DRAM interleaved",
        what);
    TT_FATAL(
        t.layout() == Layout::ROW_MAJOR && t.dtype() == dtype, "block_select: {} {} {}", what, t.dtype(), t.layout());
}

// Bytes of one strip's 32 rows (the output rows a strip covers).
constexpr uint32_t kStripRows = tt::constants::TILE_HEIGHT;
// Blocks one 512-byte input page holds, and the output run granularity.
constexpr uint32_t kBlocksPerPage = 8;
constexpr uint32_t kStripInputWidth = kBlocksPerPage * kStripRows;
constexpr uint32_t kMaxFactor = 16;

}  // namespace

void validate_inputs(const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, bool has_n) {
    const Tensor& in = t.input;
    if (attrs.strip_major) {
        TT_FATAL(
            attrs.factor == 1,
            "block_max_bf16 strip_major: factor {} (the maxes are already 32-key blocks)",
            attrs.factor);
        check_dram_rm(in, "block_max", DataType::BFLOAT16);
        TT_FATAL(
            in.logical_shape()[-1] == kStripInputWidth,
            "block_max_bf16 strip_major: width {} (pages of 8 x 32)",
            in.logical_shape()[-1]);
        TT_FATAL(t.output.has_value(), "block_max_bf16 strip_major: output_tensor [.., 32 S, NB] gives the geometry");
        TT_FATAL(has_n != t.n_tensor.has_value(), "block_max_bf16: exactly one of n / n_tensor");
        const Tensor& out = *t.output;
        check_dram_rm(out, "output", DataType::BFLOAT16);
        TT_FATAL(out.logical_shape().rank() >= 2 && in.logical_shape().rank() >= 2, "block_max_bf16 strip_major: rank");
        const uint32_t nb = out.logical_shape()[-1];
        const uint32_t s = out.logical_shape()[-2] / kStripRows;
        TT_FATAL(
            nb % kBlocksPerPage == 0 && out.logical_shape()[-2] == kStripRows * s,
            "block_max_bf16 strip_major: output {}",
            out.logical_shape());
        TT_FATAL(
            in.logical_shape()[-2] == s * nb / kBlocksPerPage,
            "block_max_bf16 strip_major: input rows {} != S {} x NB {} / 8",
            in.logical_shape()[-2],
            s,
            nb);
        return;
    }
    const uint32_t f = attrs.factor;
    TT_FATAL(f >= 1 && f <= kMaxFactor && (f & (f - 1)) == 0, "block_max_bf16: factor {}", f);
    check_dram_rm(in, "block_max", DataType::FLOAT32);
    const uint32_t ni = in.logical_shape()[-1];
    TT_FATAL(ni % (kBlocksPerPage * f) == 0, "block_max_bf16: width {} must be a multiple of 8 x factor {}", ni, f);
    TT_FATAL(has_n != t.n_tensor.has_value(), "block_max_bf16: exactly one of n / n_tensor");
    if (t.output.has_value()) {
        const Tensor& o = *t.output;
        const auto want = BlockMaxBf16DeviceOperation::compute_output_specs(attrs, {in, t.n_tensor, std::nullopt});
        TT_FATAL(
            o.dtype() == DataType::BFLOAT16 && o.layout() == Layout::ROW_MAJOR &&
                o.logical_shape() == want.logical_shape(),
            "block_max_bf16: output_tensor must be BFLOAT16 ROW_MAJOR {}, got {} {} {}",
            want.logical_shape(),
            o.dtype(),
            o.layout(),
            o.logical_shape());
        check_dram_rm(o, "output", DataType::BFLOAT16);
    } else {
        // The allocated output takes memory_config.
        const auto& mc = attrs.output_memory_config;
        TT_FATAL(
            mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "block_select: output must be DRAM interleaved");
    }
}

BlockMaxBf16DeviceOperation::program_factory_t BlockMaxBf16DeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    if (attrs.strip_major) {
        return BlockMaxStripProgramFactory{};
    }
    return BlockMaxReduceProgramFactory{};
}

void BlockMaxBf16DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // n vs n_tensor was checked by the entry, which knows whether n was given; n_tensor presence is in the key.
    validate_inputs(attrs, t, !t.n_tensor.has_value());
}

void BlockMaxBf16DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*t*/) {}

ttsl::hash::hash_t BlockMaxBf16DeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // attrs.n is left out: the factories' override_runtime_arguments re-applies it to common_arg::N on every hit.
    // The output enters as the Python key had it, by shape and memory config whether allocated or the caller's
    // (dtype and layout are pinned by validation), so an allocated output and a caller's of the same placement
    // share one program; output_memory_config only matters through it (it is unused with output_tensor).
    // Every other attribute and every input spec is hashed.
    ttnn::Shape out_shape = t.input.logical_shape();
    if (t.output.has_value()) {
        out_shape = t.output->logical_shape();
    } else {
        out_shape[-1] = out_shape[-1] / attrs.factor;
    }
    const auto& out_mc = t.output.has_value() ? t.output->memory_config() : attrs.output_memory_config;
    auto hash = tt::tt_metal::operation::hash_operation<BlockMaxBf16DeviceOperation>(
        attrs.strip_major,
        attrs.factor,
        attrs.max_grid,
        out_shape,
        out_mc,
        t.input.tensor_spec(),
        t.n_tensor.has_value());
    if (t.n_tensor.has_value()) {
        hash = ttsl::hash::hash_objects(hash, t.n_tensor->tensor_spec());
    }
    return hash;
}

BlockMaxBf16DeviceOperation::spec_return_value_t BlockMaxBf16DeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return t.output->tensor_spec();
    }
    ttnn::Shape shape = t.input.logical_shape();
    shape[-1] = shape[-1] / attrs.factor;
    return TensorSpec(
        shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attrs.output_memory_config));
}

BlockMaxBf16DeviceOperation::tensor_return_value_t BlockMaxBf16DeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input.device());
}

}  // namespace ttnn::operations::block_max_bf16

namespace ttnn::prim {

Tensor block_max_bf16(
    const Tensor& input,
    const std::optional<Tensor>& n_tensor,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_max_bf16::BlockMaxBf16Params& params) {
    using OperationType = ttnn::operations::block_max_bf16::BlockMaxBf16DeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .n_tensor = n_tensor, .output = output});
}

}  // namespace ttnn::prim
