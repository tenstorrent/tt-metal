// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp"

using namespace tt::tt_metal;

ttnn::Shape squeeze_vector_shape(ttnn::Shape output_shape) {
    if (output_shape.rank() > 4) {
        ttnn::SmallVector<uint32_t> output_shape_4d(output_shape.rank());
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
    BaseUntilizeValType base_untilize, const std::optional<CoreRangeSet>& sub_core_grids) {
    auto original_shape = std::make_shared<Shape>();

    return MassagedUntilizeVal(MassagedUntilizeValParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeValArgs {
            *original_shape = input_tensor.logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor, sub_core_grids);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(
                output,
                *original_shape,
                std::nullopt,              /*Memory Config*/
                std::nullopt,              /*Pad value*/
                TileReshapeMapMode::CACHE, /*Reshape map mode*/
                sub_core_grids);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

ttnn::Tensor ExecuteUntilizeWithUnpadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_tensor_end,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    bool use_pack_untilize,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool fp32_dest_acc_en = input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::FLOAT32;

    ttnn::SmallVector<uint32_t> output_end_vector;
    ttnn::Shape output_end;
    const auto& input_shape = input_tensor.logical_shape();
    if (input_shape.rank() > 4) {
        for (auto index = 0; index < input_shape.rank(); ++index) {
            output_end_vector.push_back(input_shape[index] - 1);
        }
        output_end = squeeze_vector_shape(ttnn::Shape(std::move(output_end_vector)));
    } else {
        for (auto index = 0; index < input_tensor.logical_shape().rank(); ++index) {
            output_end_vector.push_back(output_tensor_end[index]);
        }
        output_end = ttnn::Shape(std::move(output_end_vector));
    }

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size = input_single_tile_size;

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_per_col = input_tensor.padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    bool enough_space_width =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_untilize = [=](const ttnn::Tensor& input_tensor) {
        return ttnn::prim::untilize_with_unpadding(
            input_tensor,
            ttnn::Shape(output_end),
            memory_config,
            use_multicore,
            use_pack_untilize,
            fp32_dest_acc_en,
            enough_space_width,
            enough_space_height,
            sub_core_grids);
    };

    return build_ndiml_untilize_val(base_untilize, sub_core_grids)(input_tensor);
}

}  // namespace ttnn::operations::data_movement
