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
    // CollectSpecData rejects the spec ("DFB has no producer"). The device operation's validation
    // and output spec are both fine on an empty input, so run them and allocate the result - the
    // only thing skipped is the program, which cannot be built with no work to do.
    if (input_tensor.logical_volume() == 0) {
        // create_device_tensor dereferences the device, which is null for a host tensor.
        TT_FATAL(
            input_tensor.device() != nullptr, "untilize_with_unpadding: input tensor must be allocated on a device");

        // output_tensor_end holds inclusive end indices, so each extent is end + 1; for an empty dim
        // that end is the uint32 wrap of 0 - 1, and + 1 returns it to 0. Built over the input's rank
        // so the result comes back at the caller's rank, as the ndiml wrapper would deliver it.
        ttsl::SmallVector<uint32_t> empty_shape;
        empty_shape.reserve(input_shape.rank());
        for (size_t index = 0; index < input_shape.rank(); ++index) {
            empty_shape.push_back(output_tensor_end[index] + 1);
        }
        const ttnn::Shape output_shape(std::move(empty_shape));
        // validate_on_program_cache_miss never looks at output_tensor_end, so these two have no
        // counterpart there. Unpadding only shrinks, and volume alone misses a grown axis:
        // [0, 64] with ends [0, UINT32_MAX] is zero-volume but shaped [1, 0].
        TT_FATAL(
            output_shape.volume() == 0,
            "untilize_with_unpadding: a zero-volume input requires a zero-volume output, got {}",
            output_shape);
        for (size_t index = 0; index < input_shape.rank(); ++index) {
            TT_FATAL(
                output_shape[index] <= input_tensor.padded_shape()[index],
                "untilize_with_unpadding: output extent {} exceeds the padded input extent {} in "
                "dimension {}",
                output_shape[index],
                input_tensor.padded_shape()[index],
                index);
        }

        // Everything else is the device operation's to decide. fp32_dest_acc_en and
        // enough_space_height only steer factory selection, which is skipped, and neither
        // validate nor compute_output_specs reads them.
        const ttnn::prim::UntilizeWithUnpaddingParams attributes{
            .output_tensor_end = ttnn::Shape(ttsl::SmallVector<uint32_t>(
                output_tensor_end.cbegin(), output_tensor_end.cbegin() + input_shape.rank())),
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .use_multicore = use_multicore,
            .fp32_dest_acc_en = false,
            .enough_space_height = false,
            .sub_core_grids = sub_core_grids,
        };
        ttnn::prim::UntilizeWithUnpaddingDeviceOperation::validate_on_program_cache_miss(attributes, input_tensor);
        // The one rule that lives in select_program_factory rather than validate.
        TT_FATAL(
            !input_tensor.is_sharded() || !sub_core_grids.has_value(),
            "Sharded untilize does not support sub core grid specification");

        // Allocated rather than filled: there is no element to initialise, and going through a host
        // tensor would upload to the device, which fails inside trace capture and drops the input's
        // mesh topology on the way.
        return create_device_tensor(
            ttnn::prim::UntilizeWithUnpaddingDeviceOperation::compute_output_specs(attributes, input_tensor),
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
