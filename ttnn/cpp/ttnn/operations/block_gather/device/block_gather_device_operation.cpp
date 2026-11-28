// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_gather_device_operation.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::block_gather {

using namespace tt::tt_metal;

namespace {

bool is_dram_interleaved(const MemoryConfig& mc) {
    return mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

// Python _dram_rm.
void check_dram_rm(const Tensor& t, const char* what, DataType dtype) {
    TT_FATAL(
        is_dram_interleaved(t.memory_config()),
        "block_select: {} must be DRAM interleaved, got {}",
        what,
        t.memory_config());
    TT_FATAL(
        t.layout() == Layout::ROW_MAJOR && t.dtype() == dtype, "block_select: {} {} {}", what, t.dtype(), t.layout());
}

}  // namespace

void validate_inputs(const BlockGatherParams& attrs, const BlockGatherInputs& t) {
    const uint32_t b = attrs.block;
    TT_FATAL(b >= 1 && b <= 32 && (b & (b - 1)) == 0, "block_gather: block {} must be a power of two <= 32", b);
    check_dram_rm(t.scores, "scores", DataType::BFLOAT16);
    check_dram_rm(t.ids, "ids", DataType::UINT32);
    const auto& ss = t.scores.logical_shape();
    const auto& is = t.ids.logical_shape();
    const uint32_t w = ss[-1];
    TT_FATAL(
        w % tt::constants::TILE_WIDTH == 0 && w % b == 0,
        "block_gather: width {} must be a multiple of 32 and of the block {}",
        w,
        b);
    bool same_lead = ss.rank() == is.rank();
    for (int32_t i = 0; same_lead && i + 1 < static_cast<int32_t>(ss.rank()); ++i) {
        same_lead = ss[i] == is[i];
    }
    TT_FATAL(same_lead, "block_gather: leading dims differ: {} {}", ss, is);
    const auto spec = BlockGatherDeviceOperation::compute_output_specs(attrs, t);
    if (t.output.has_value()) {
        const Tensor& o = *t.output;
        TT_FATAL(
            o.dtype() == DataType::BFLOAT16 && o.layout() == Layout::ROW_MAJOR &&
                o.logical_shape() == spec.logical_shape(),
            "block_select block_gather: output_tensor must be BFLOAT16 ROW_MAJOR {}, got {} {} {}",
            spec.logical_shape(),
            o.dtype(),
            o.layout(),
            o.logical_shape());
        check_dram_rm(o, "output", DataType::BFLOAT16);
    } else {
        TT_FATAL(
            is_dram_interleaved(attrs.output_memory_config),
            "block_select: output must be DRAM interleaved, got {}",
            attrs.output_memory_config);
    }
}

BlockGatherDeviceOperation::program_factory_t BlockGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return BlockGatherProgramFactory{};
}

void BlockGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_inputs(attrs, tensor_args);
}

void BlockGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t BlockGatherDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The program depends on the block, the core rectangle, the scores and ids specs (W, K, rows, their accessors)
    // and the output's placement (its accessor); the output's dtype, layout and shape follow from those (a caller's
    // output of any other is refused). Whether the output is the caller's is not structural: the override writes its
    // address on a hit, so both share one program, as the Python's prepared program did. attrs.output_memory_config
    // is not hashed when a caller's output is given, since the call ignores it then.
    const MemoryConfig& out_mc =
        tensor_args.output.has_value() ? tensor_args.output->memory_config() : attrs.output_memory_config;
    return tt::tt_metal::operation::hash_operation<BlockGatherDeviceOperation>(
        attrs.block,
        attrs.grid_x,
        attrs.grid_y,
        out_mc,
        tensor_args.scores.tensor_spec(),
        tensor_args.ids.tensor_spec());
}

BlockGatherDeviceOperation::spec_return_value_t BlockGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    ttnn::Shape shape = tensor_args.scores.logical_shape();
    shape[-1] = tensor_args.ids.logical_shape()[-1] * attrs.block;
    return TensorSpec(
        shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attrs.output_memory_config));
}

BlockGatherDeviceOperation::tensor_return_value_t BlockGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.scores.device());
}

}  // namespace ttnn::operations::block_gather

namespace ttnn::prim {

Tensor block_gather(
    const Tensor& scores,
    const Tensor& ids,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_gather::BlockGatherParams& params) {
    using OperationType = ttnn::operations::block_gather::BlockGatherDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.scores = scores, .ids = ids, .output = output});
}

}  // namespace ttnn::prim
