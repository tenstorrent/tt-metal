// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include "device/untilize_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
using OwnedUntilizeArgs = std::tuple<ttnn::Tensor>;
using BaseUntilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedUntilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedUntilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedUntilize build_ndiml_untilize(BaseUntilizeType base_untilize) {
    auto original_shape = std::make_shared<std::pair<ttnn::Shape, ttnn::Shape>>();
    return MassagedUntilize(MassagedUntilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeArgs {
            *original_shape = std::make_pair(input_tensor.logical_shape(), input_tensor.padded_shape());
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(output, original_shape->first, original_shape->second);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

ttnn::Tensor ExecuteUntilize::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    bool use_pack_untilize,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool fp32_dest_acc_en = input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::FLOAT32;

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
        auto pf_type = ttnn::operations::data_movement::get_pf_type(
            memory_config.has_value() ? memory_config.value().is_sharded() : input_tensor.is_sharded(), input_tensor);

        return ttnn::prim::untilize(
            input_tensor,
            memory_config.value_or(input_tensor.memory_config()),
            use_multicore,
            use_pack_untilize,
            fp32_dest_acc_en,
            sub_core_grids,
            enough_space_width,
            enough_space_height,
            pf_type);
    };

    return build_ndiml_untilize(base_untilize)(input_tensor);
}

}  // namespace ttnn::operations::data_movement
