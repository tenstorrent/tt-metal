// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include <fmt/core.h>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::quasar {

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    if (op_attr.is_sharded) {
        return MultiCore{};
    }
    return MultiCoreDRAMFold{};
}

void validate_fold(const std::vector<Tensor>& input_tensors, bool is_sharded, uint32_t stride_h, uint32_t stride_w) {
    const Tensor& input_tensor = input_tensors.at(0);
    const auto& logical_shape = input_tensor.logical_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");
    // Guard before any modulo/div on stride; both sharded and unsharded branches SIGFPE on zero stride.
    TT_FATAL(stride_h > 0 && stride_w > 0, "Fold: stride_h ({}) and stride_w ({}) must be > 0.", stride_h, stride_w);
    if (is_sharded) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Fold: Only height-sharded input tensors are supported.");
        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (logical_shape[2] * stride_h) == 0,
            "Fold: Shard height must be divisible by input width times stride_h for proper folding operation.");
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Fold: Expect sharded input tensor in row-major layout.");
    } else {
        // Divisibility on logical (padded hides partial-tile W that the tile-native writer would OOB into).
        TT_FATAL(
            logical_shape[1] % stride_h == 0,
            "Fold: logical H ({}) must be divisible by stride_h ({}).",
            logical_shape[1],
            stride_h);
        TT_FATAL(
            logical_shape[2] % stride_w == 0,
            "Fold: logical W ({}) must be divisible by stride_w ({}).",
            logical_shape[2],
            stride_w);
        // Composite falls back before prim, so this only reaches direct prim::qsr::fold callers.
        if (input_tensor.layout() == tt::tt_metal::Layout::TILE) {
            auto reason = tile_native_fold_rejection_reason(input_tensor, stride_h, stride_w);
            TT_FATAL(
                !reason.has_value(),
                "Fold (TILE): tile-native gate refused: {}; untilize input to RM first.",
                reason.value_or(""));
        }
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_sharded, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_sharded, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    // launch calls compute_output_specs before validate; unguarded input_shape / (stride_h * stride_w) SIGFPEs.
    TT_FATAL(
        op_attr.stride_h > 0 && op_attr.stride_w > 0,
        "Fold: stride_h ({}) and stride_w ({}) must be > 0.",
        op_attr.stride_h,
        op_attr.stride_w);
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    const tt::tt_metal::DataType output_dtype = fold_output_dtype(input_tensor.dtype());

    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    ttnn::Shape output_shape(
        {1,
         1,
         input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_sharded) {
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
        shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        auto mem_config = MemoryConfig(
            input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), shard_spec);

        return {tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config))};
    }
    // Interleaved DRAM keeps the folded 4D shape (both TILE and RM factories land RM).
    ttnn::Shape output_logical_shape = output_shape;
    if (input_tensor.memory_config().is_dram()) {
        output_logical_shape = ttnn::Shape(
            {input_shape[0],
             input_shape[1] / op_attr.stride_h,
             input_shape[2] / op_attr.stride_w,
             input_shape[3] * op_attr.stride_h * op_attr.stride_w});
    }
    return {tt::tt_metal::TensorSpec(
        output_logical_shape,
        tt::tt_metal::TensorLayout(
            output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), input_tensor.memory_config()))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {
ttnn::operations::experimental::quasar::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    using OperationType = ttnn::operations::experimental::quasar::Fold;
    auto operation_attributes = OperationType::operation_attributes_t{
        .stride_h = stride_h, .stride_w = stride_w, .is_sharded = input_tensor.is_sharded()};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim::qsr
