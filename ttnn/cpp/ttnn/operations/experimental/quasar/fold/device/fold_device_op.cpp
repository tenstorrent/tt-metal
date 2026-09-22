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

// hal::get_max_worker_l1_unreserved_size() is defined as l1_end - KERNEL_CONFIG_addr, i.e. the
// budget *if the ringbuffer size is 0*. DFB allocations actually start at DEFAULT_UNRESERVED_addr,
// past the KERNEL_CONFIG region (~105 KB on WH, similar on BH). Reserve covers that gap + JIT
// dataflow/compute code+stack so the predicate stays a strict upper bound on real CB alloc.
static constexpr uint64_t kFoldL1CodeStackReserveBytes = 160 * 1024;

std::optional<std::string> tile_native_fold_rejection_reason(
    const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    // Zero-stride short-circuit: the composite consults this predicate before prim::qsr::fold, so
    // divide-by-zero in tile_native_fold_scratch_bytes (input_width / stride_w) would beat
    // validate_fold's stride > 0 guard and turn a clean FATAL into a host SIGFPE.
    if (stride_h == 0 || stride_w == 0) {
        return fmt::format("stride_h={} or stride_w={} (must be > 0)", stride_h, stride_w);
    }
    if (input_tensor.layout() != tt::tt_metal::Layout::TILE) {
        return std::string{"non-TILE layout"};
    }
    if (input_tensor.is_sharded()) {
        return std::string{"sharded input"};
    }
    // Writer's tt_memmove only hits the NoC self-copy path with 16B-aligned c_bytes; otherwise it
    // falls into per-pixel CPU memmove and loses ~2x on small C (composite routes those to RM).
    const uint32_t out_elem = tt::datum_size(datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype())));
    const uint32_t c_bytes = input_tensor.logical_shape()[-1] * out_elem;
    if (c_bytes % 16 != 0) {
        return fmt::format("c_bytes={} (not 16B-aligned)", c_bytes);
    }
    const uint64_t scratch = tile_native_fold_scratch_bytes(input_tensor, stride_h, stride_w);
    const auto in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto out_df = datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype()));
    const uint32_t c_tiles = tt::div_up(input_tensor.padded_shape()[-1], tt::constants::TILE_WIDTH);
    // Match the factory: SRC0 and SRC1 each carry kFoldSrcCbDepthPerCTile × c_tiles entries so
    // untilize/gather can pipeline. Predicate scales with the same constant so routing tracks alloc.
    const uint64_t cb_bytes =
        static_cast<uint64_t>(tt::tile_size(in_df) + tt::tile_size(out_df)) * c_tiles * kFoldSrcCbDepthPerCTile;
    // Static arch budget minus this op's own CB reservations + code/stack; keeps routing a pure
    // function of inputs (not live allocator state). Any real fragmentation still surfaces at CB alloc.
    const uint64_t budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
    if (scratch + cb_bytes + kFoldL1CodeStackReserveBytes >= budget) {
        return fmt::format("scratch={} B + CBs={} B exceed L1 budget ({} B)", scratch, cb_bytes, budget);
    }
    return std::nullopt;
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
    // Guard both branches before any modulo/div on stride; sharded branch does % (W * stride_h) and
    // % stride_h below, unsharded does % stride_h / % stride_w — both SIGFPE on zero stride.
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
        TT_FATAL(logical_shape[1] % stride_h == 0, "Fold: logical H must be divisible by stride_h.");
        TT_FATAL(logical_shape[2] % stride_w == 0, "Fold: logical W must be divisible by stride_w.");
        // Tile-native gate: one FATAL, one source of truth (the predicate). Composite consults the
        // same predicate and falls back to untilize→RM before prim, so this only reaches direct
        // prim::qsr::fold callers. Reason string carries the distinguishing substring.
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
    // Same stride > 0 guard as validate_fold: launch calls compute_output_specs before validate,
    // so an unguarded input_shape / (stride_h * stride_w) below would SIGFPE and skip the FATAL.
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
