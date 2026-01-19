// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding.hpp"

#include "device/tilize_with_val_padding_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

using OwnedTilizeValArgs = std::tuple<ttnn::Tensor>;
using BaseTilizeValType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedTilizeVal = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedTilizeValParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedTilizeVal build_ndiml_tilize_val(
    BaseTilizeValType base_tilize, const std::optional<CoreRangeSet>& sub_core_grids) {
    auto original_shape = std::make_shared<Shape>();
    return MassagedTilizeVal(MassagedTilizeValParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedTilizeValArgs {
            *original_shape = input_tensor.logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor, sub_core_grids);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(
                output,
                *original_shape,
                std::nullopt /*Memory Config*/,
                std::nullopt /*pad value*/,
                TileReshapeMapMode::CACHE,
                sub_core_grids);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_tilize)});
}

ttnn::Shape squeeze_output_shape(const ttnn::Shape& output_shape) {
    if (output_shape.rank() > 4) {
        std::array<uint32_t, 4> output_shape_4d{};
        output_shape_4d[0] = 1;
        int extra_rank = output_shape.rank() - 4;
        for (int i = extra_rank; i >= 0; i--) {
            output_shape_4d[0] *= output_shape[i];
        }
        output_shape_4d[1] = output_shape[1 + extra_rank];
        output_shape_4d[2] = output_shape[2 + extra_rank];
        output_shape_4d[3] = output_shape[3 + extra_rank];
        return ttnn::Shape(output_shape_4d);
    }
    return output_shape;
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (input_tensor.layout() == Layout::TILE) {
        return input_tensor;
    }

    // Handle empty tensors - no tiling needed for tensors with no data
    if (input_tensor.physical_volume() == 0) {
        // Create output tensor with same properties
        TensorSpec spec(
            output_padded_shape,
            TensorLayout(
                output_dtype.value_or(input_tensor.dtype()),
                PageConfig(Layout::TILE),
                memory_config.value_or(input_tensor.memory_config())));
        return create_device_tensor(spec, input_tensor.device());
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value() ? tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
                                 : input_single_tile_size;

    uint32_t num_tiles_per_row = output_padded_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_per_col = output_padded_shape[-2] / tt::constants::TILE_HEIGHT;

    bool enough_space_width =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        return ttnn::prim::tilize_with_val_padding(
            input_tensor,
            squeeze_output_shape(output_padded_shape),
            pad_value,
            memory_config.value_or(input_tensor.memory_config()),
            output_dtype.value_or(input_tensor.dtype()),
            use_multicore,
            enough_space_width,
            enough_space_height,
            sub_core_grids);
    };

    return build_ndiml_tilize_val(base_tilize, sub_core_grids)(input_tensor);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Handle empty tensors - no tiling needed for tensors with no data
    if (input_tensor.physical_volume() == 0) {
        // Create output tensor with same properties
        TensorSpec spec(
            ttnn::Shape{output_padded_shape},
            TensorLayout(
                output_dtype.value_or(input_tensor.dtype()),
                PageConfig(Layout::TILE),
                memory_config.value_or(input_tensor.memory_config())));
        return create_device_tensor(spec, input_tensor.device());
    }

    return invoke(
        input_tensor,
        ttnn::Shape{output_padded_shape},
        pad_value,
        memory_config,
        output_dtype,
        use_multicore,
        sub_core_grids);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace tt::constants;
    auto padded_shape = input_tensor.padded_shape();

    uint32_t input_tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t input_tile_height = input_tensor.tensor_spec().tile().get_height();

    padded_shape[-2] = tt::round_up(padded_shape[-2], input_tile_height);
    padded_shape[-1] = tt::round_up(padded_shape[-1], input_tile_width);

    // Handle empty tensors - no tiling needed for tensors with no data
    if (input_tensor.physical_volume() == 0) {
        // Create output tensor with same properties
        TensorSpec spec(
            padded_shape,
            TensorLayout(
                output_dtype.value_or(input_tensor.dtype()),
                PageConfig(Layout::TILE),
                memory_config.value_or(input_tensor.memory_config())));
        return create_device_tensor(spec, input_tensor.device());
    }

    PadValue pad_value;
    if (input_tensor.dtype() == DataType::BFLOAT16 or input_tensor.dtype() == DataType::FLOAT32) {
        pad_value = 0.0f;
    } else {
        pad_value = (uint32_t)0;
    }
    return ExecuteTilizeWithValPadding::invoke(
        input_tensor, padded_shape, pad_value, memory_config, output_dtype, use_multicore, sub_core_grids);
}

}  // namespace ttnn::operations::data_movement
