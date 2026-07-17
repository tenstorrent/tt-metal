// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "device/tilize_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::quasar {
// Shared data-movement helpers (data_movement/common/common.hpp, reshape_view/reshape.hpp) used
// bare by this op; they previously resolved via the enclosing data_movement namespace.
using ttnn::operations::data_movement::MassagedOperation;
using ttnn::operations::data_movement::MassagedOperationParams;
using ttnn::operations::data_movement::squeeze_from_ND_to_4D;
using ttnn::operations::experimental::quasar::TileReshapeMapMode;

using OwnedTilizeArgs = std::tuple<ttnn::Tensor>;
using BaseTilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedTilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedTilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedTilize build_ndiml_tilize(BaseTilizeType base_tilize, const std::optional<CoreRangeSet>& sub_core_grids) {
    auto original_shape = std::make_shared<Shape>();
    return MassagedTilize(MassagedTilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedTilizeArgs {
            *original_shape = input_tensor.logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor, sub_core_grids);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::operations::experimental::quasar::reshape(
                output, *original_shape, std::nullopt, std::nullopt, TileReshapeMapMode::CACHE, sub_core_grids);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_tilize)});
}

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    bool use_low_perf,
    const std::optional<CoreRangeSet>& sub_core_grids,
    tt::tt_metal::Tile tile) {
    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            input_tensor.tensor_spec().tile() == tile,
            "ttnn::experimental::quasar::tilize: TILE tensor already uses tile {}, cannot reinterpret as {}",
            input_tensor.tensor_spec().tile(),
            tile);
        TT_FATAL(
            !memory_config.has_value() || memory_config.value() == input_tensor.memory_config(),
            "ttnn::experimental::quasar::tilize: cannot silently drop requested memory_config on already-TILE input");
        TT_FATAL(
            !output_dtype.has_value() || output_dtype.value() == input_tensor.dtype(),
            "ttnn::experimental::quasar::tilize: cannot silently drop requested dtype on already-TILE input");
        return input_tensor;
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value() ? tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
                                 : input_single_tile_size;

    uint32_t input_tile_width = tile.get_width();
    uint32_t input_tile_height = tile.get_height();

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / input_tile_width;
    uint32_t num_tiles_per_col = input_tensor.padded_shape()[-2] / input_tile_height;

    bool enough_space_width = ttnn::operations::data_movement::is_enough_space(
        input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height = ttnn::operations::data_movement::is_enough_space(
        input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        return ttnn::prim::qsr::tilize(
            input_tensor,
            memory_config,
            output_dtype,
            use_multicore,
            enough_space_width,
            enough_space_height,
            use_low_perf,
            sub_core_grids,
            tile);
    };

    return ttnn::operations::experimental::quasar::build_ndiml_tilize(base_tilize, sub_core_grids)(input_tensor);
}

}  // namespace ttnn::operations::experimental::quasar
