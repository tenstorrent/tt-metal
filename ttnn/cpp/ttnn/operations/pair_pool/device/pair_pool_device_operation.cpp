// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pair_pool_device_operation.hpp"

#include <utility>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::pair_pool {

using namespace tt::tt_metal;

namespace {

// The message is formatted only when the check fails: the entry runs these checks on every call.
template <typename... Args>
void refuse_unless(bool ok, fmt::format_string<Args...> format, Args&&... args) {
    if (!ok) {
        throw PairPoolAssertion(fmt::format(format, std::forward<Args>(args)...));
    }
}

ttnn::Shape pooled_shape(const Tensor& x) {
    ttnn::Shape shape = x.logical_shape();
    shape[-2] /= 2;
    shape[-1] /= 2;
    return shape;
}

}  // namespace

void check_support(const PairPoolParams& /*attrs*/, const PairPoolInputs& t) {
    const Tensor& x = t.input;
    refuse_unless(
        x.dtype() == DataType::FLOAT32 && x.layout() == Layout::TILE,
        "(DataType.{}, Layout.{})",
        enchantum::to_string(x.dtype()),
        enchantum::to_string(x.layout()));
    const MemoryConfig& mc = x.memory_config();
    refuse_unless(
        mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED, "{}", mc);
    const uint32_t w = x.logical_shape()[-1];
    refuse_unless(w % 64 == 0, "pair_pool: last dim {} must be 2 hd with hd % 32 == 0", w);
    const uint32_t rows = x.logical_shape()[-2];
    refuse_unless(rows % 64 == 0 && x.padded_shape()[-2] == rows, "pair_pool: rows {} % 64 != 0", rows);
    if (t.output.has_value()) {
        const Tensor& o = *t.output;
        refuse_unless(
            o.dtype() == DataType::FLOAT32 && o.layout() == Layout::TILE && o.logical_shape() == pooled_shape(x),
            "(DataType.{}, {})",
            enchantum::to_string(o.dtype()),
            o.logical_shape());
        refuse_unless(o.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "{}", o.memory_config());
    }
}

PairPoolDeviceOperation::program_factory_t PairPoolDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return PairPoolProgramFactory{};
}

void PairPoolDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    check_support(attrs, tensor_args);
}

void PairPoolDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t PairPoolDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Every attribute (all structural) and the input's spec. The caller's output_tensor, present or absent, is left
    // out, as the Python op's program key (mesh, x padded shape, memory configs, max_grid) left it out: the entry
    // refuses one whose dtype, layout or shape differs from the output spec, and its memory config is
    // attrs.output_memory_config, so with or without it the program is the same; override_runtime_arguments
    // re-applies its address (the writer's OUT_ADDR) on every hit.
    return tt::tt_metal::operation::hash_operation<PairPoolDeviceOperation>(attrs, tensor_args.input.tensor_spec());
}

PairPoolDeviceOperation::spec_return_value_t PairPoolDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return TensorSpec(
        pooled_shape(tensor_args.input),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_memory_config));
}

PairPoolDeviceOperation::tensor_return_value_t PairPoolDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::pair_pool

namespace ttnn::prim {

Tensor pair_pool(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const ttnn::operations::pair_pool::PairPoolParams& params) {
    using OperationType = ttnn::operations::pair_pool::PairPoolDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .output = output});
}

}  // namespace ttnn::prim
