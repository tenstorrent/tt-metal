// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp"

using namespace tt::tt_metal;

ttnn::Shape squeeze_vector_shape(ttnn::Shape output_shape) {
    if (output_shape.rank() > 4) {
        ttsl::SmallVector<uint32_t> output_shape_4d(4);
        output_shape_4d[0] = 1;
        int extra_rank = output_shape.size() - 4;
        for (int i = extra_rank; i >= 0; i--) {
            output_shape_4d[0] *= (output_shape[i] + 1);
        }
        output_shape_4d[0]--;
        output_shape_4d[1] = output_shape[1 + extra_rank];
        output_shape_4d[2] = output_shape[2 + extra_rank];
        output_shape_4d[3] = output_shape[3 + extra_rank];
        return ttnn::Shape(std::move(output_shape_4d));
    }
    return output_shape;
}

namespace ttnn::operations::data_movement {

using OwnedUntilizeValArgs = std::tuple<ttnn::Tensor>;
using BaseUntilizeValType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedUntilizeVal = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedUntilizeValParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedUntilizeVal build_ndiml_untilize_val(
    BaseUntilizeValType base_untilize,
    const Shape& output_tensor_end,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto output_shape = std::make_shared<Shape>();

    return MassagedUntilizeVal(MassagedUntilizeValParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeValArgs {
            ttsl::SmallVector<uint32_t> output_shape_vector;
            output_shape_vector.reserve(output_tensor_end.rank());
            for (auto index = 0; index < output_tensor_end.rank(); ++index) {
                output_shape_vector.push_back(output_tensor_end[index] + 1);
            }
            *output_shape = ttnn::Shape(std::move(output_shape_vector));
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor, sub_core_grids);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(
                output,
                *output_shape,
                std::nullopt,              /*Memory Config*/
                std::nullopt,              /*Pad value*/
                TileReshapeMapMode::CACHE, /*Reshape map mode*/
                sub_core_grids);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

}  // namespace ttnn::operations::data_movement

namespace ttnn {

Tensor untilize_with_unpadding(
    const Tensor& input_tensor,
    const Shape& output_tensor_end,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool fp32_dest_acc_en = input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                            input_tensor.dtype() == DataType::FLOAT32;

    ttsl::SmallVector<uint32_t> output_end_vector;
    ttnn::Shape output_end;
    const auto& input_shape = input_tensor.logical_shape();
    for (auto index = 0; index < input_shape.rank(); ++index) {
        output_end_vector.push_back(output_tensor_end[index]);
    }

    if (input_shape.rank() > 4) {
        output_end = squeeze_vector_shape(ttnn::Shape(std::move(output_end_vector)));
    } else {
        output_end = ttnn::Shape(std::move(output_end_vector));
    }

    // The prim emits BFLOAT16 for a BFLOAT8_B input (UntilizeWithUnpaddingDeviceOperation::
    // compute_output_specs), so size the output CB estimate and the pending output buffer by the output
    // dtype, as ttnn::untilize does. Sized by the input dtype, the output CB estimate was a 1088 B tile
    // against the row factory's 2048 B/tile output CB, and the reservation was 0 B: a BFLOAT8_B ROW_MAJOR
    // TensorSpec cannot be constructed ("Only TILE layout is supported for BFLOAT8_B dtype"), which
    // get_pending_l1_output_reservation waves through as nothing to reserve. Both made enough_space_height
    // more permissive than the row factory it selects.
    const DataType output_dtype = operations::data_movement::untilize_output_dtype(input_tensor.dtype());

    // Nothing to untilize. The factories split work by block count, which is 0 for an empty input,
    // so no WorkUnitSpec is emitted while the dataflow buffers are already declared, and
    // CollectSpecData rejects the spec ("DFB has no producer"). Allocate the output here instead.
    //
    // This stands in for the device operation's validate and compute_output_specs, so it repeats
    // the checks they would have made and derives the output shard the way they would. The rule is:
    // never accept a request the normal path rejects, and never try to support one it cannot do.
    if (input_tensor.logical_volume() == 0) {
        // create_device_tensor dereferences the device, which is null for a host tensor.
        TT_FATAL(
            input_tensor.device() != nullptr, "untilize_with_unpadding: input tensor must be allocated on a device");
        TT_FATAL(input_tensor.layout() == Layout::TILE, "Can only untilize tile major data");
        TT_FATAL(
            !input_tensor.is_sharded() || !sub_core_grids.has_value(),
            "Sharded untilize does not support sub core grid specification");

        // output_tensor_end holds inclusive end indices, so each extent is end + 1; for an empty
        // dim that end is the uint32 wrap of 0 - 1, and + 1 returns it to 0. Built over the input's
        // rank, since output_tensor_end may be longer and taking its rank would grow an axis.
        ttsl::SmallVector<uint32_t> empty_shape;
        empty_shape.reserve(input_shape.rank());
        for (size_t index = 0; index < input_shape.rank(); ++index) {
            empty_shape.push_back(output_tensor_end[index] + 1);
        }
        const ttnn::Shape output_shape(std::move(empty_shape));
        TT_FATAL(
            output_shape.volume() == 0,
            "untilize_with_unpadding: a zero-volume input requires a zero-volume output, got {}",
            output_shape);
        // Unpadding only shrinks. Volume alone misses that: [0, 64] with ends [0, UINT32_MAX] is
        // zero-volume but shaped [1, 0].
        for (size_t index = 0; index < input_shape.rank(); ++index) {
            TT_FATAL(
                output_shape[index] <= input_tensor.padded_shape()[index],
                "untilize_with_unpadding: output extent {} exceeds the padded input extent {} in "
                "dimension {}",
                output_shape[index],
                input_tensor.padded_shape()[index],
                index);
        }

        auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
        const auto input_layout = input_tensor.memory_config().memory_layout();
        // The single-core implementation cannot write an output shard narrower than a tile.
        if (!use_multicore && output_mem_config.is_sharded()) {
            TT_FATAL(
                output_mem_config.shard_spec().has_value() || output_mem_config.nd_shard_spec().has_value(),
                "Output memory config is sharded but no shard spec or nd shard spec is provided");
            const uint32_t output_shard_width = output_mem_config.shard_spec().has_value()
                                                    ? output_mem_config.shard_spec().value().shape[1]
                                                    : output_mem_config.nd_shard_spec().value().shard_shape[-1];
            TT_FATAL(
                output_shard_width % tt::constants::TILE_WIDTH == 0,
                "Output shard width {} must be a multiple of tile width {} for single core implementation",
                output_shard_width,
                tt::constants::TILE_WIDTH);
        }
        const bool input_is_sharded = input_tensor.shard_spec().has_value();
        // A caller may name a sharded layout without a spec (ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        // and friends); the device operation fills it in from the input for a same-layout output.
        if (input_is_sharded && output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value() &&
            output_mem_config.memory_layout() == input_layout) {
            output_mem_config = MemoryConfig(
                output_mem_config.memory_layout(), output_mem_config.buffer_type(), input_tensor.shard_spec());
        }

        // Which sharded output each sharded input may produce. TensorSpec checks the output's own
        // geometry, not whether the conversion is supported. A same-layout output binds the output
        // buffer as a dynamic circular buffer, which only works in L1.
        const bool same_layout = input_is_sharded && output_mem_config.memory_layout() == input_layout;
        if (input_is_sharded && output_mem_config.is_sharded()) {
            if (input_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                TT_FATAL(
                    same_layout,
                    "Output memory config layout must be HEIGHT_SHARDED when output is sharded but got {}",
                    output_mem_config.memory_layout());
            } else if (input_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    same_layout || output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                    "Output memory config layout ({}) must be BLOCK_SHARDED (or WIDTH_SHARDED with a "
                    "matching column shard width) when input is BLOCK_SHARDED and output is sharded",
                    output_mem_config.memory_layout());
            } else if (input_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                TT_FATAL(
                    same_layout || output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
                    "Output memory config layout ({}) must match input tensor memory layout ({}) (or be "
                    "BLOCK_SHARDED with a matching column shard width)",
                    output_mem_config.memory_layout(),
                    input_layout);
            }
            if (same_layout) {
                TT_FATAL(
                    output_mem_config.buffer_type() == BufferType::L1,
                    "{} -> {} output must be in L1; got buffer_type={}",
                    input_layout,
                    output_mem_config.memory_layout(),
                    output_mem_config.buffer_type());
            }
            // The device operation's remaining same-type check is an unbatched-only assert that
            // divides by padded_shape[-2] * padded_shape[-1]. That is 0 here, and "batch" has no
            // meaning without elements, so it has no counterpart on this path.
        }

        // The output shard shape, derived as the device operation derives it. Its fused height is
        // volume() / shape[-1], which is 0 / 0 for an empty output, so use the product of the
        // leading dims - the same value whenever the width is non-zero. A derived extent of 0 means
        // that axis is empty, and no shard shape can hold a 0, so the base extent is kept there.
        if (output_mem_config.is_sharded() && output_mem_config.shard_spec().has_value()) {
            uint32_t fused_height = 1;
            for (size_t index = 0; index + 1 < output_shape.rank(); ++index) {
                fused_height *= output_shape[index];
            }
            const auto tile = input_tensor.tensor_spec().tile();
            // A same-layout output is reshaped from the input's shard, as the device operation does;
            // a cross-layout one keeps the caller's verbatim, and an interleaved input has none to
            // inherit, so its extents come from the grid.
            ShardSpec output_shard_spec =
                same_layout ? input_tensor.shard_spec().value() : output_mem_config.shard_spec().value();
            const CoreRange bbox = output_shard_spec.grid.bounding_box();
            uint32_t grid_cols = bbox.end_coord.x - bbox.start_coord.x + 1;
            uint32_t grid_rows = bbox.end_coord.y - bbox.start_coord.y + 1;
            if (output_shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
                std::swap(grid_cols, grid_rows);
            }
            const uint32_t num_cores = output_shard_spec.num_cores();
            const uint32_t base_height = output_shard_spec.shape[0];
            const uint32_t base_width = output_shard_spec.shape[1];
            const auto or_base = [](uint32_t derived, uint32_t base) { return derived > 0 ? derived : base; };
            const auto out_layout = output_mem_config.memory_layout();
            if (!input_tensor.memory_config().is_sharded()) {
                if (out_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                    output_shard_spec.shape = {
                        or_base(fused_height, base_height),
                        or_base(tt::round_up(tt::div_up(output_shape[-1], num_cores), tile.get_width()), base_width)};
                } else if (out_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                    output_shard_spec.shape = {
                        or_base(tt::round_up(tt::div_up(fused_height, grid_rows), tile.get_height()), base_height),
                        or_base(tt::round_up(tt::div_up(output_shape[-1], grid_cols), tile.get_width()), base_width)};
                } else {
                    output_shard_spec.shape = {
                        or_base(tt::round_up(tt::div_up(fused_height, num_cores), tile.get_height()), base_height),
                        base_width};
                }
            } else if (same_layout) {
                if (out_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                    output_shard_spec.shape = {or_base(fused_height, base_height), base_width};
                } else if (out_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                    output_shard_spec.shape = {
                        or_base(tt::round_up(tt::div_up(fused_height, grid_rows), tile.get_height()), base_height),
                        or_base(tt::round_up(tt::div_up(output_shape[-1], grid_cols), tile.get_width()), base_width)};
                } else {
                    output_shard_spec.shape = {
                        or_base(tt::round_up(tt::div_up(fused_height, num_cores), tile.get_height()), base_height),
                        or_base(output_shape[-1], base_width)};
                }
            }
            output_mem_config =
                MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), output_shard_spec);
        }
        // Allocated rather than filled: there is no element to initialise, and going through a host
        // tensor would upload to the device, which fails outright inside trace capture and drops the
        // input's mesh topology on the way. Carry that topology across instead.
        return create_device_tensor(
            tt::tt_metal::TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), output_mem_config)),
            input_tensor.device(),
            input_tensor.tensor_topology());
    }

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Reserve the output buffer's per-core L1 up front: it is allocated after this check
    // but before the CBs are placed, so leaving it out overestimates the CB budget and can
    // pick a factory whose static CBs then clash with it.
    // output_end holds inclusive end indices, so the produced shape is end + 1 per dim.
    ttsl::SmallVector<uint32_t> output_shape_vector;
    output_shape_vector.reserve(output_end.rank());
    for (size_t index = 0; index < output_end.rank(); ++index) {
        output_shape_vector.push_back(output_end[index] + 1);
    }
    const uint32_t pending_l1_output_bytes = operations::data_movement::get_pending_l1_output_reservation(
        input_tensor,
        ttnn::Shape(std::move(output_shape_vector)),
        memory_config.value_or(input_tensor.memory_config()),
        output_dtype,
        Layout::ROW_MAJOR);

    bool enough_space_height = operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_row,
        /*staging_bytes_per_tile=*/0,
        /*fixed_staging_bytes=*/0,
        pending_l1_output_bytes);

    auto base_untilize = [=](const ttnn::Tensor& input_tensor) {
        return ttnn::prim::untilize_with_unpadding(
            input_tensor,
            ttnn::Shape(output_end),
            memory_config,
            use_multicore,
            fp32_dest_acc_en,
            enough_space_height,
            sub_core_grids);
    };

    return ttnn::operations::data_movement::build_ndiml_untilize_val(
        base_untilize, output_tensor_end, sub_core_grids)(input_tensor);
}

}  // namespace ttnn
