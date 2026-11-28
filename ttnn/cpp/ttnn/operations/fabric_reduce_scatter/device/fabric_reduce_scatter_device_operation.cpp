// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fabric_reduce_scatter_device_operation.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

using namespace tt::tt_metal;

namespace {

// (Sb, W) of one output block.
std::pair<uint32_t, uint32_t> block_dims(const FabricReduceScatterParams& attrs, const Tensor& input) {
    const auto& shape = input.logical_shape();
    TT_FATAL(attrs.group_size >= 2, "fabric_reduce_scatter: a group needs at least two chips");
    if (attrs.dim == 2) {
        return {shape[2] / attrs.group_size, shape[3]};
    }
    return {shape[2], shape[3] / attrs.group_size};
}

TensorSpec dram_tile_spec(const Tensor& input, uint32_t rows, uint32_t cols) {
    return TensorSpec(
        Shape({1, 1, rows, cols}), TensorLayout(input.dtype(), PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
}

// Whether two device tensors share memory: the same buffer type and intersecting per-bank address ranges (a view
// wraps the same address in a Buffer of its own, so Buffer pointers do not tell).
bool shares_memory(const Tensor& a, const Tensor& b) {
    const Buffer* x = a.buffer();
    const Buffer* y = b.buffer();
    if (x->buffer_type() != y->buffer_type()) {
        return false;
    }
    const uint64_t x_begin = x->address();  // smuggled-rta-ok: overlap check, not a kernel argument
    const uint64_t y_begin = y->address();  // smuggled-rta-ok: overlap check, not a kernel argument
    return x_begin < y_begin + y->aligned_size_per_bank() && y_begin < x_begin + x->aligned_size_per_bank();
}

// What the key leaves out and every call (hits included) must hold: the scratch and a preallocated output on the
// input's mesh, and the scratch sharing memory with neither the input nor the output.
void check_placement(const FabricReduceScatterInputs& tensor_args) {
    TT_FATAL(tensor_args.scratch.has_value(), "fabric_reduce_scatter: scratch_tensor is required");
    const Tensor& scratch = *tensor_args.scratch;
    TT_FATAL(
        scratch.device() == tensor_args.input.device(),
        "fabric_reduce_scatter: scratch_tensor must be on the input's mesh");
    TT_FATAL(
        !shares_memory(scratch, tensor_args.input),
        "fabric_reduce_scatter: scratch_tensor shares memory with the input");
    if (tensor_args.output.has_value()) {
        TT_FATAL(
            tensor_args.output->device() == tensor_args.input.device(),
            "fabric_reduce_scatter: output must be on the input's mesh");
        TT_FATAL(
            !shares_memory(scratch, *tensor_args.output),
            "fabric_reduce_scatter: scratch_tensor shares memory with the output");
    }
}

}  // namespace

FabricReduceScatterDeviceOperation::program_factory_t FabricReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return FabricReduceScatterMeshWorkloadFactory{};
}

void FabricReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.dim == 2 || attrs.dim == 3, "fabric_reduce_scatter: dim {} must be 2 or 3", attrs.dim);
    check_placement(tensor_args);
    const Tensor& scratch = *tensor_args.scratch;
    TT_FATAL(
        scratch.tensor_spec() == scratch_spec(attrs, tensor_args.input),
        "fabric_reduce_scatter: scratch_tensor spec {} does not match {}",
        scratch.tensor_spec(),
        scratch_spec(attrs, tensor_args.input));
}

void FabricReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    check_placement(tensor_args);
}

ttsl::hash::hash_t FabricReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Every attribute and the specs of input, scratch and output (the given one's, else the one the op allocates).
    // Left out: whether the output was preallocated. The kernels write either through the same accessor, so a
    // preallocated output of the allocated spec shares the program; the override re-applies its address.
    TT_FATAL(tensor_args.scratch.has_value(), "fabric_reduce_scatter: scratch_tensor is required");
    return tt::tt_metal::operation::hash_operation<FabricReduceScatterDeviceOperation>(
        attrs,
        tensor_args.input.tensor_spec(),
        tensor_args.scratch->tensor_spec(),
        compute_output_specs(attrs, tensor_args));
}

TensorSpec FabricReduceScatterDeviceOperation::scratch_spec(const operation_attributes_t& attrs, const Tensor& input) {
    const auto [sb, w] = block_dims(attrs, input);
    return dram_tile_spec(input, (attrs.group_size + 1) * sb, w);
}

FabricReduceScatterDeviceOperation::spec_return_value_t FabricReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }
    const auto [sb, w] = block_dims(attrs, tensor_args.input);
    return dram_tile_spec(tensor_args.input, sb, w);
}

FabricReduceScatterDeviceOperation::tensor_return_value_t FabricReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::fabric_reduce_scatter

namespace ttnn::prim {

Tensor fabric_reduce_scatter(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::fabric_reduce_scatter::FabricReduceScatterParams& params) {
    using OperationType = ttnn::operations::fabric_reduce_scatter::FabricReduceScatterDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .output = output, .scratch = scratch});
}

}  // namespace ttnn::prim
