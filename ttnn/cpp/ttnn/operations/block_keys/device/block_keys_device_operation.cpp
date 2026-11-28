// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_keys_device_operation.hpp"

#include <string_view>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::block_keys {

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& mc) {
    return mc.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

// Python _dram_rm(t, what, uint32).
void check_dram_rm_u32(const Tensor& t, std::string_view what) {
    TT_FATAL(
        is_dram_interleaved(t.memory_config()),
        "block_select: {} must be DRAM interleaved, got {}",
        what,
        t.memory_config());
    TT_FATAL(
        t.layout() == tt::tt_metal::Layout::ROW_MAJOR && t.dtype() == tt::tt_metal::DataType::UINT32,
        "block_select: {} {} {}",
        what,
        t.dtype(),
        t.layout());
}

}  // namespace

void validate_inputs(const BlockKeysParams& attrs, const BlockKeysInputs& t) {
    const uint32_t b = attrs.block;
    TT_FATAL(b >= 1 && b <= 32 && (b & (b - 1)) == 0, "block_keys: block {} must be a power of two <= 32", b);
    check_dram_rm_u32(t.pos, "pos");
    check_dram_rm_u32(t.ids, "ids");
    const auto& ps = t.pos.logical_shape();
    const auto& is = t.ids.logical_shape();
    bool same_prefix = ps.rank() == is.rank();
    for (int i = 0; same_prefix && i + 1 < static_cast<int>(ps.rank()); ++i) {
        same_prefix = ps[i] == is[i];
    }
    TT_FATAL(same_prefix, "block_keys: pos {} and ids {} must share every dim but the last", ps, is);
    if (t.output.has_value()) {
        const Tensor& o = *t.output;
        TT_FATAL(
            o.dtype() == tt::tt_metal::DataType::UINT32 && o.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                o.logical_shape() == ps,
            "block_select block_keys: output_tensor must be UINT32 ROW_MAJOR {}, got {} {} {}",
            ps,
            o.dtype(),
            o.layout(),
            o.logical_shape());
    }
    TT_FATAL(
        is_dram_interleaved(attrs.output_memory_config),
        "block_select: output must be DRAM interleaved, got {}",
        attrs.output_memory_config);
}

BlockKeysDeviceOperation::program_factory_t BlockKeysDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return BlockKeysProgramFactory{};
}

void BlockKeysDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_inputs(attrs, tensor_args);
}

void BlockKeysDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t BlockKeysDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Every structural input: the block, width / block, the core rectangle, the output's placement (its accessor;
    // a caller's output gives attrs.output_memory_config, see the public entry) and the pos / ids specs (KP, K, the
    // rows, their accessors). The output's dtype, layout and shape follow from those (a caller's output of any other
    // is refused). Whether the output is the caller's is left out: the override writes its address on a hit, so both
    // share one program, as the Python's prepared program (keyed on shapes and memory configs) did.
    return tt::tt_metal::operation::hash_operation<BlockKeysDeviceOperation>(
        attrs.block,
        attrs.nbw,
        attrs.grid_x,
        attrs.grid_y,
        attrs.output_memory_config,
        tensor_args.pos.tensor_spec(),
        tensor_args.ids.tensor_spec());
}

BlockKeysDeviceOperation::spec_return_value_t BlockKeysDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Python _out: pos's shape with the last dim KP (pos's own), uint32 ROW_MAJOR.
    return tt::tt_metal::TensorSpec(
        tensor_args.pos.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_memory_config));
}

BlockKeysDeviceOperation::tensor_return_value_t BlockKeysDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.pos.device());
}

}  // namespace ttnn::operations::block_keys

namespace ttnn::prim {

Tensor block_keys(
    const Tensor& pos,
    const Tensor& ids,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_keys::BlockKeysParams& params) {
    using OperationType = ttnn::operations::block_keys::BlockKeysDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.pos = pos, .ids = ids, .output = output});
}

}  // namespace ttnn::prim
