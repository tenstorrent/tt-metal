// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_tail_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

namespace fused_tail_op_detail {

void check_dram_interleaved(const Tensor& t, const char* what, tt::tt_metal::Layout layout) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "fused_tail: {} must be on device", what);
    TT_FATAL(t.buffer() != nullptr, "fused_tail: {} is not allocated", what);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "fused_tail: {} must be bfloat16, got {}", what, t.dtype());
    TT_FATAL(t.layout() == layout, "fused_tail: {} must be {}, got {}", what, layout, t.layout());
    TT_FATAL(
        t.memory_config().buffer_type() == BufferType::DRAM &&
            t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "fused_tail: {} must be DRAM interleaved",
        what);
}

void check_shape(const Tensor& t, const char* what, uint32_t rows, uint32_t cols) {
    const auto& s = t.logical_shape();
    TT_FATAL(
        s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] == rows && s[3] == cols,
        "fused_tail: {} must be [1, 1, {}, {}], got {}",
        what,
        rows,
        cols,
        s);
}

}  // namespace fused_tail_op_detail
using namespace fused_tail_op_detail;

ttsl::hash::hash_t FusedTailDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto spec_of = [](const std::optional<Tensor>& t) {
        return t.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t->tensor_spec()) : std::nullopt;
    };
    return tt::tt_metal::operation::hash_operation<FusedTailDeviceOperation>(
        attrs.kind,
        attrs.plan_key,
        attrs.math_fidelity,
        attrs.nslot,
        attrs.output_memory_config,
        args.y_own.tensor_spec(),
        spec_of(args.shared_expert),
        spec_of(args.w_up));
}

FusedTailDeviceOperation::program_factory_t FusedTailDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    if (attrs.kind == TailKind::Engine) {
        return TailEngineFactory{};
    }
    return TailA2AFactory{};
}

void FusedTailDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    check_dram_interleaved(args.y_own, "y_own", Layout::ROW_MAJOR);
    if (attrs.kind == TailKind::Engine) {
        TT_FATAL(attrs.engine_plan != nullptr, "fused_tail: the engine plan is missing");
        const auto& g = attrs.engine_plan->geom;
        check_shape(args.y_own, "y_own", g.rows / g.chunks, g.Hr);
        TT_FATAL(args.shared_expert.has_value(), "fused_tail: a latent model's tail needs the shared expert's output");
        check_dram_interleaved(*args.shared_expert, "shared_expert", Layout::TILE);
        check_shape(*args.shared_expert, "shared_expert", g.rows, g.Ho);
        TT_FATAL(args.w_up.has_value(), "fused_tail: a latent model's tail needs w_up (prepare_fused_tail_weights)");
        TT_FATAL(
            args.w_up->buffer() != nullptr && args.w_up->memory_config().buffer_type() == BufferType::DRAM,
            "fused_tail: w_up must be a DRAM tensor of prepare_fused_tail_weights");
    } else {
        TT_FATAL(attrs.a2a_plan != nullptr, "fused_tail: the a2a plan is missing");
        const auto& g = attrs.a2a_plan->geom;
        check_shape(args.y_own, "y_own", g.S, g.H);
        if (args.shared_expert.has_value()) {
            check_dram_interleaved(*args.shared_expert, "shared_expert", Layout::TILE);
            check_shape(*args.shared_expert, "shared_expert", g.S, g.H);
        }
        TT_FATAL(!args.w_up.has_value(), "fused_tail: a model without a latent takes no w_up");
        TT_FATAL(
            args.arrival.has_value() && args.staging.has_value(), "fused_tail: the a2a's persistent words are missing");
    }
    if (args.output.has_value()) {
        TT_FATAL(
            args.output->tensor_spec() == compute_output_specs(attrs, args),
            "fused_tail: the output must be {}",
            compute_output_specs(attrs, args));
    }
}

void FusedTailDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

FusedTailDeviceOperation::spec_return_value_t FusedTailDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    uint32_t rows = 0, cols = 0;
    if (attrs.kind == TailKind::Engine) {
        rows = attrs.engine_plan->geom.rows;
        cols = attrs.engine_plan->geom.Ho;
    } else {
        rows = attrs.a2a_plan->geom.rows;
        cols = attrs.a2a_plan->geom.H / attrs.a2a_plan->geom.G;
    }
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, rows, cols}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_memory_config));
}

FusedTailDeviceOperation::tensor_return_value_t FusedTailDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    if (args.output.has_value()) {
        return *args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, args), args.y_own.device());
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail

namespace ttnn::prim {

Tensor moe_block_fused_tail(
    const ttnn::operations::experimental::moe_block::fused_tail::FusedTailParams& params,
    ttnn::operations::experimental::moe_block::fused_tail::FusedTailInputs inputs) {
    using OperationType = ttnn::operations::experimental::moe_block::fused_tail::FusedTailDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, std::move(inputs));
}

}  // namespace ttnn::prim
