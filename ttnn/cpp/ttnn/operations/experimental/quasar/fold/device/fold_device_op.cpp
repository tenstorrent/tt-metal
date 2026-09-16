// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::quasar {

tt::tt_metal::DataType fold_output_dtype(tt::tt_metal::DataType input_dtype) {
    return (input_dtype == tt::tt_metal::DataType::FLOAT32 || input_dtype == tt::tt_metal::DataType::UINT16)
               ? input_dtype
               : tt::tt_metal::DataType::BFLOAT16;
}

uint64_t tile_native_fold_scratch_bytes(const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    const uint32_t out_elem = tt::datum_size(datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype())));
    const auto& shape = input_tensor.logical_shape();
    const uint32_t input_width = shape[2];
    const uint32_t C = shape[-1];
    return static_cast<uint64_t>(input_width / stride_w) * stride_h * stride_w * C * out_elem;
}

bool tile_native_fold_scratch_fits_l1(const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    if (input_tensor.layout() != tt::tt_metal::Layout::TILE) {
        return false;
    }
    const uint64_t scratch = tile_native_fold_scratch_bytes(input_tensor, stride_h, stride_w);
    const auto in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto out_df = datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype()));
    const uint32_t c_tiles = tt::div_up(input_tensor.padded_shape()[-1], tt::constants::TILE_WIDTH);
    const uint64_t cb_bytes = static_cast<uint64_t>(tt::tile_size(in_df) + tt::tile_size(out_df)) * c_tiles;
    constexpr uint64_t kCodeStackReserve = 32 * 1024;
    return scratch + cb_bytes + kCodeStackReserve < tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
}

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
        TT_FATAL(logical_shape[1] % stride_h == 0, "Fold: logical H must be divisible by stride_h.");
        TT_FATAL(logical_shape[2] % stride_w == 0, "Fold: logical W must be divisible by stride_w.");
        // TILE input routes through the tile-native factory; refuse configs whose row scratch won't fit L1.
        TT_FATAL(
            input_tensor.layout() != tt::tt_metal::Layout::TILE ||
                tile_native_fold_scratch_fits_l1(input_tensor, stride_h, stride_w),
            "Fold (TILE): tile-native scratch {} B + tile CBs exceed per-core L1; untilize input to RM first.",
            tile_native_fold_scratch_bytes(input_tensor, stride_h, stride_w));
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
    // Interleaved tensors (DRAM or L1): DRAM keeps the folded 4D shape (both TILE and RM go through
    // fold_multi_core_tiled_interleaved / fold_multi_core_row_major_interleaved and land RM).
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
