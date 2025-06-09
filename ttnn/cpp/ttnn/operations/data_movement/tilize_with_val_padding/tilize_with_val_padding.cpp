// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding.hpp"

#include "device/tilize_with_val_padding_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

using OwnedTilizeValArgs = std::tuple<ttnn::Tensor>;
using BaseTilizeValType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedTilizeVal = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedTilizeValParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedTilizeVal build_ndiml_tilize_val(BaseTilizeValType base_tilize) {
    auto original_shape = std::make_shared<Shape>();
    return MassagedTilizeVal(MassagedTilizeValParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedTilizeValArgs {
            *original_shape = input_tensor.logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            const auto tile = output.tensor_spec().tile();
            uint32_t tile_height = tile.get_height();
            uint32_t tile_width = tile.get_width();
            auto unsqueezed_tensor = ttnn::reshape(output, *original_shape);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_tilize)});
}

ttnn::Shape squeeze_output_shape(const ttnn::Shape& output_shape) {
    if (output_shape.rank() > 4) {
        std::array<uint32_t, 4> output_shape_4d;
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
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value()
            ? tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
            : input_single_tile_size;

    uint32_t num_tiles_per_row = output_padded_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_per_col = output_padded_shape[-2] / tt::constants::TILE_HEIGHT;

    bool enough_space_width =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        return operation::run(
            TilizeWithValPadding{
                squeeze_output_shape(output_padded_shape),
                pad_value,
                memory_config.value_or(input_tensor.memory_config()),
                output_dtype.value_or(input_tensor.dtype()),
                use_multicore,
                enough_space_width,
                enough_space_height},
            {input_tensor},
            {},
            {},
            queue_id)[0];
    };

    return build_ndiml_tilize_val(base_tilize)(input_tensor);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        DefaultQueueId, input_tensor, output_padded_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        queue_id,
        input_tensor,
        ttnn::Shape{output_padded_shape},
        pad_value,
        memory_config,
        output_dtype,
        use_multicore);
}

ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(
        DefaultQueueId, input_tensor, output_padded_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    using namespace tt::constants;
    auto padded_shape = input_tensor.padded_shape();

    uint32_t input_tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t input_tile_height = input_tensor.tensor_spec().tile().get_height();

    padded_shape[-2] = tt::round_up(padded_shape[-2], input_tile_height);
    padded_shape[-1] = tt::round_up(padded_shape[-1], input_tile_width);

    PadValue pad_value;
    if (input_tensor.dtype() == DataType::BFLOAT16 or input_tensor.dtype() == DataType::FLOAT32) {
        pad_value = 0.0f;
    } else {
        pad_value = (uint32_t)0;
    }
    return ExecuteTilizeWithValPadding::invoke(
        queue_id, input_tensor, padded_shape, pad_value, memory_config, output_dtype, use_multicore);
}

ttnn::Tensor ExecuteTilizeWithZeroPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
}

}  // namespace ttnn::operations::data_movement
