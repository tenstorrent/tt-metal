// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_tail_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_op_detail {

void check_dram_interleaved(const Tensor& tensor, const char* what, tt::tt_metal::Layout layout) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "moe_tail: {} must be on device", what);
    TT_FATAL(tensor.buffer() != nullptr, "moe_tail: {} is not allocated", what);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "moe_tail: {} must be bfloat16, got {}", what, tensor.dtype());
    TT_FATAL(tensor.layout() == layout, "moe_tail: {} must be {}, got {}", what, layout, tensor.layout());
    TT_FATAL(
        tensor.memory_config().buffer_type() == BufferType::DRAM &&
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "moe_tail: {} must be DRAM interleaved",
        what);
}

void check_shape(const Tensor& tensor, const char* what, uint32_t rows, uint32_t columns) {
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == rows && shape[3] == columns,
        "moe_tail: {} must be [1, 1, {}, {}], got {}",
        what,
        rows,
        columns,
        shape);
}

}  // namespace moe_tail_op_detail
using namespace moe_tail_op_detail;

ttsl::hash::hash_t MoETailDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    const auto spec_of = [](const std::optional<Tensor>& tensor) {
        return tensor.has_value() ? std::optional<tt::tt_metal::TensorSpec>(tensor->tensor_spec()) : std::nullopt;
    };
    return tt::tt_metal::operation::hash_operation<MoETailDeviceOperation>(
        attributes.kind,
        attributes.plan_key,
        attributes.math_fidelity,
        attributes.ring_slots,
        attributes.row_blocks,
        attributes.first_block,
        attributes.forward_chunks,
        attributes.row_tokens,
        args.actual_start.has_value(),
        args.metadata.has_value(),
        attributes.output_memory_config,
        args.y_own.tensor_spec(),
        spec_of(args.shared_expert),
        spec_of(args.w_up));
}

MoETailDeviceOperation::program_factory_t MoETailDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t&) {
    if (attributes.kind == TailKind::Engine) {
        return TailEngineFactory{};
    }
    return TailAllToAllFactory{};
}

void MoETailDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    check_dram_interleaved(args.y_own, "y_own", Layout::ROW_MAJOR);
    if (attributes.kind == TailKind::Engine) {
        TT_FATAL(attributes.engine_plan != nullptr, "moe_tail: the engine plan is missing");
        const auto& geometry = attributes.engine_plan->geometry;
        check_shape(
            args.y_own, "y_own", geometry.rows / geometry.num_chunks * attributes.row_blocks, geometry.latent_dim);
        TT_FATAL(args.shared_expert.has_value(), "moe_tail: a latent model's tail needs the shared expert's output");
        check_dram_interleaved(*args.shared_expert, "shared_expert", Layout::TILE);
        check_shape(*args.shared_expert, "shared_expert", geometry.rows * attributes.row_blocks, geometry.output_dim);
        TT_FATAL(args.w_up.has_value(), "moe_tail: a latent model's tail needs w_up (prepare_moe_tail_weights)");
        TT_FATAL(
            args.w_up->buffer() != nullptr && args.w_up->memory_config().buffer_type() == BufferType::DRAM,
            "moe_tail: w_up must be a DRAM tensor of prepare_moe_tail_weights");
    } else {
        TT_FATAL(attributes.all_to_all_plan != nullptr, "moe_tail: the all-to-all plan is missing");
        const auto& geometry = attributes.all_to_all_plan->geometry;
        check_shape(args.y_own, "y_own", geometry.owned_rows * attributes.row_blocks, geometry.hidden_dim);
        if (args.shared_expert.has_value()) {
            check_dram_interleaved(*args.shared_expert, "shared_expert", Layout::TILE);
            check_shape(
                *args.shared_expert, "shared_expert", geometry.owned_rows * attributes.row_blocks, geometry.hidden_dim);
        }
        TT_FATAL(!args.w_up.has_value(), "moe_tail: a model without a latent takes no w_up");
        TT_FATAL(
            args.arrival.has_value() && args.staging.has_value(),
            "moe_tail: the all-to-all's persistent words are missing");
    }
    if (args.output.has_value()) {
        TT_FATAL(
            args.output->tensor_spec() == compute_output_specs(attributes, args),
            "moe_tail: the output must be {}",
            compute_output_specs(attributes, args));
    }
}

void MoETailDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    validate_on_program_cache_miss(attributes, args);
}

MoETailDeviceOperation::spec_return_value_t MoETailDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t&) {
    uint32_t rows = 0, columns = 0;
    if (attributes.kind == TailKind::Engine) {
        rows = attributes.engine_plan->geometry.rows;
        columns = attributes.engine_plan->geometry.output_dim;
    } else {
        rows = attributes.all_to_all_plan->geometry.rows;
        columns = attributes.all_to_all_plan->geometry.hidden_dim / attributes.all_to_all_plan->geometry.tp_size;
    }
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, rows * attributes.row_blocks, columns}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), attributes.output_memory_config));
}

MoETailDeviceOperation::tensor_return_value_t MoETailDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    if (args.output.has_value()) {
        return *args.output;
    }
    return create_device_tensor(compute_output_specs(attributes, args), args.y_own.device());
}

}  // namespace ttnn::operations::experimental::moe_block::moe_tail

namespace ttnn::prim {

Tensor moe_block_tail(
    const ttnn::operations::experimental::moe_block::moe_tail::MoETailParams& params,
    ttnn::operations::experimental::moe_block::moe_tail::MoETailInputs inputs) {
    using OperationType = ttnn::operations::experimental::moe_block::moe_tail::MoETailDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, std::move(inputs));
}

}  // namespace ttnn::prim
