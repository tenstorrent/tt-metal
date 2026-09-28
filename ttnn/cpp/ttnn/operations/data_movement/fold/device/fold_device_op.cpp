// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/synthesize_output_shard_spec.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement {

bool is_fast_path_input(const Tensor& t) {
    return t.memory_config().is_l1() && t.is_sharded() && t.shard_spec().has_value() &&
           t.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && t.layout() == Layout::ROW_MAJOR;
}

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
    // src0/src1 tile CBs scale with C_tiles (the factory sizes both to C_tiles entries).
    const auto in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto out_df = datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype()));
    const uint32_t c_tiles = tt::div_up(input_tensor.padded_shape()[-1], tt::constants::TILE_WIDTH);
    const uint64_t cb_bytes = static_cast<uint64_t>(tt::tile_size(in_df) + tt::tile_size(out_df)) * c_tiles;
    constexpr uint64_t kCodeStackReserve = 32 * 1024;
    return scratch + cb_bytes + kCodeStackReserve < tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
}

tt::tt_metal::ShardSpec synthesize_fold_output_shard_spec(
    const Tensor& input_tensor, tt::tt_metal::TensorMemoryLayout layout, uint32_t rows, uint32_t cols) {
    // COL_MAJOR input now yields column-wise H/W CoreRangeSet (documented via fold COL_MAJOR H/W shrink test).
    auto* device = input_tensor.device();
    return common::synthesize_output_shard_spec(
        device->compute_with_storage_grid_size(),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols),
        layout,
        {.is_tile = false,
         .input_orientation = input_tensor.shard_spec().has_value()
                                  ? std::optional{input_tensor.shard_spec()->orientation}
                                  : std::nullopt,
         .caller_tag = "Fold"});
}

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& tensors) {
    return is_fast_path_input(tensors.input_tensor) ? program_factory_t{MultiCore{}}
                                                    : program_factory_t{MultiCoreDRAMFold{}};
}

void validate_fold(const std::vector<Tensor>& input_tensors, uint32_t stride_h, uint32_t stride_w) {
    const Tensor& input_tensor = input_tensors.at(0);
    const auto& logical_shape = input_tensor.logical_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");

    // Reject zero strides before any modulo/div; guards both fast + composite paths and compute_output_specs.
    TT_FATAL(stride_h > 0 && stride_w > 0, "Fold: stride_h ({}) and stride_w ({}) must be > 0.", stride_h, stride_w);
    // Divisibility on logical (padded hides partial-tile W remainders that the tile-native writer would OOB into).
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

    // TILE input routes through the tile-native factory (in prim::fold); refuse configs whose row scratch won't fit L1.
    // Composite falls back to untilize→RM before prim, so this only fires on direct prim::fold(TILE) callers.
    TT_FATAL(
        input_tensor.layout() != tt::tt_metal::Layout::TILE ||
            tile_native_fold_scratch_fits_l1(input_tensor, stride_h, stride_w),
        "Fold (TILE): tile-native scratch {} B + tile CBs exceed per-core L1; untilize input to RM first.",
        tile_native_fold_scratch_bytes(input_tensor, stride_h, stride_w));

    if (is_fast_path_input(input_tensor)) {
        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (logical_shape[2] * stride_h) == 0,
            "Fold (fast path): shard height must be divisible by input width times stride_h.");
    } else if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value()) {
        // Per-shard sticks must be patch_size-divisible; else fold silently truncates. Specless → compute_output_specs
        // synthesises.
        const auto& spec = input_tensor.shard_spec().value();
        const uint32_t patch_size = stride_h * stride_w;
        TT_FATAL(
            spec.shape[0] % patch_size == 0,
            "Fold: sharded input shard height ({}) must be divisible by patch_size ({}).",
            spec.shape[0],
            patch_size);
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    const auto& input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    const tt::tt_metal::DataType output_dtype = fold_output_dtype(input_tensor.dtype());

    // NHWC pixel_unshuffle; folded_4d vs collapsed picked upfront — shard bytes identical.
    const uint32_t out_N = input_shape[0];
    const uint32_t out_H = input_shape[1] / op_attr.stride_h;
    const uint32_t out_W = input_shape[2] / op_attr.stride_w;
    const uint32_t out_C = input_shape[3] * op_attr.stride_h * op_attr.stride_w;
    ttnn::Shape output_shape = op_attr.collapse_output ? ttnn::Shape({1, 1, out_N * out_H * out_W, out_C})
                                                       : ttnn::Shape({out_N, out_H, out_W, out_C});

    tt::tt_metal::MemoryConfig out_mem_config = input_tensor.memory_config();
    if (input_tensor.is_sharded()) {
        const uint32_t patch_size = op_attr.stride_h * op_attr.stride_w;
        // Concrete spec → rescale (h/=patch, w*=patch); specless → synthesise via shared helper.
        tt::tt_metal::ShardSpec output_spec =
            input_tensor.shard_spec().has_value()
                ? [&] {
                      auto s = input_tensor.shard_spec().value();
                      s.shape[0] /= patch_size;
                      s.shape[1] *= patch_size;
                      return s;
                  }()
                : synthesize_fold_output_shard_spec(input_tensor, input_tensor.memory_config().memory_layout(), out_N * out_H * out_W, out_C);
        out_mem_config = MemoryConfig(
            input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), output_spec);
    }

    // Single output-shape rule — folded_4d (or collapsed) for every mem_cfg/layout; composite (fold.cpp) handles any
    // TILE-input post-reshape it needs from the fixed contract.
    return {tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), out_mem_config))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w, bool collapse_output) {
    using OperationType = ttnn::operations::data_movement::Fold;
    auto operation_attributes = OperationType::operation_attributes_t{
        .stride_h = stride_h, .stride_w = stride_w, .collapse_output = collapse_output};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
