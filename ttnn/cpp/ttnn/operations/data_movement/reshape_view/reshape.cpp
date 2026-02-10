// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "reshape.hpp"
#include "reshape_common.hpp"
#include "device/reshape_device_operation.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

MemoryConfig recompute_shard_spec_for_output(const MemoryConfig& memory_config, const TensorSpec& output_shape) {
    // This function recomputes the shard spec as reshape op's original input
    // tensor is sometimes not compatible with the output shard's config

    auto output_mem_config = memory_config;
    if (memory_config.shard_spec().has_value()) {
        const auto& input_shard_spec = memory_config.shard_spec().value();

        // Update specs for output tensor
        auto orientation = input_shard_spec.orientation;

        if (memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            auto core_range = input_shard_spec.grid.bounding_box();
            auto updated_spec = output_shape.block_sharded(core_range, orientation);
            output_mem_config = updated_spec.memory_config();
        } else if (memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            auto core_range = input_shard_spec.grid;
            auto updated_spec = output_shape.height_sharded(core_range, orientation);
            output_mem_config = updated_spec.memory_config();
        } else if (memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            auto core_range = input_shard_spec.grid;
            auto updated_spec = output_shape.width_sharded(core_range, orientation);
            output_mem_config = updated_spec.memory_config();
        } else {
            TT_FATAL(false, "Shard spec must be either block, height, or width sharded");
        }
    } else {
        TT_FATAL(false, "Shard spec has no value");
    }
    return output_mem_config;
}

ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const MemoryConfig& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    std::cout << "PERFORMING RESHAPE ON 2D RM" << std::endl;

    if (tensor.memory_config().is_sharded() || memory_config.is_sharded()) {
        TT_FATAL(!sub_core_grid.has_value(), "Sharded reshape does not support sub core grid specification\n");
    }

    auto output_tensor =
        ttnn::prim::reshape_view(tensor, logical_shape, padded_shape, memory_config, false, sub_core_grid);

    std::cout << "OUTPUT TENSOR HAS BEEN CALCULATED" << std::endl;

    if (memory_config.is_sharded()) {
        // Recompute the shard spec for the output tensor shape
        auto output_mem_config = recompute_shard_spec_for_output(memory_config, output_tensor.tensor_spec());
        std::cout << "recompute complete" << std::endl;
        if (output_mem_config.shard_spec().value() != memory_config.shard_spec().value()) {
            if (output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                std::cout << "HEIGHT SHARDED HIT" << std::endl;
                auto shard_spec = output_mem_config.shard_spec().value();
                shard_spec.shape[1] = logical_shape[-1];
                output_mem_config = output_mem_config.with_shard_spec(shard_spec);
            } else if (output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                std::cout << "WIDTH SHARDED HIT" << std::endl;
                auto shard_spec = output_mem_config.shard_spec().value();
                shard_spec.shape[0] = logical_shape[-2];
                output_mem_config = output_mem_config.with_shard_spec(shard_spec);
            }
            std::cout << "about to reshape output " << std::endl;
            output_tensor =
                ttnn::prim::reshape_view(tensor, logical_shape, padded_shape, output_mem_config, false, sub_core_grid);
        }
    }
    std::cout << "RETURNING OUTPUT TENSOR" << std::endl;
    return output_tensor;
}

ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // This function turns a RM 2D->MD into an equivalent 2D->2D conversion and then turns the 2D output back to MD
    // using a 0 cost view
    TT_FATAL((logical_shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    // Collapse into the second last dimension
    uint32_t second_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(logical_shape.rank()) - 1; ++i) {
        second_dim = second_dim * logical_shape[i];
    }
    return PerformView(
        perform_reshape_on_2D_RM(
            tensor,
            ttnn::Shape({second_dim, logical_shape[-1]}),
            ttnn::Shape({second_dim, logical_shape[-1]}),
            memory_config,
            sub_core_grid),
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim);
}

// Wrapper to turn the ND-> MD problem into 3D->3D for tiled and 2D->2D for Row Major
ttnn::Tensor reshape_rm(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const PadValue& /*pad_value*/,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // This function turns ND -> MD into 2D->MD for row major and 3D->MD for tiled using a 0 cost view
    const auto& tensor_shape = tensor.logical_shape();
    TT_FATAL((tensor_shape.rank() != 0), "Can't do reshape from rank 0 tensor");
    TT_FATAL(tensor.layout() == ttnn::ROW_MAJOR_LAYOUT, "Wrong layout in `reshape_rm` `");

    // Collapse into the second last dimension
    uint32_t second_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(tensor_shape.rank()) - 1; ++i) {
        second_dim = second_dim * tensor_shape[i];
    }
    // Call reshape with the equivalent data 2D Row Major input tensor
    return fix_shape_and_perform_reshape_on_2D_RM(
        PerformView(
            tensor,
            Shape({second_dim, tensor_shape[-1]}),
            Shape({second_dim, tensor_shape[-1]}),
            tile_first_dim,
            tile_second_dim),
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim,
        memory_config,
        sub_core_grid);
}
}  // namespace detail

ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim = tt::constants::TILE_HEIGHT,
    const uint32_t tile_second_dim = tt::constants::TILE_WIDTH) {
    if (tensor.logical_shape() == logical_shape && tensor.padded_shape() == padded_shape) {
        return tensor;
    }
    if (logical_shape.rank() == 1) {
        return ttnn::experimental::view(tensor, logical_shape);
    }
    if (tensor.layout() == ttnn::TILE_LAYOUT &&
        (logical_shape[-1] % tile_first_dim != 0 || logical_shape[-2] % tile_second_dim != 0)) {
        return ttnn::experimental::view(tensor, logical_shape, compute_padded_shape(logical_shape));
    }
    // Perform a reshape (view)
    return ttnn::experimental::view(tensor, logical_shape, padded_shape);
}

std::pair<ttnn::Shape, ttnn::Shape> shape_corrector(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    // Correct the shape to account for inferred dimensions
    uint32_t input_volume = tensor.logical_volume();
    uint32_t output_volume = 1;
    uint32_t inferred_dim = -1;
    for (uint32_t i = 0; i < logical_shape.rank(); i++) {
        if ((static_cast<int>(logical_shape[i])) == -1) {
            if (inferred_dim != -1) {
                TT_FATAL(false, "Only one dimension can be inferred in reshape");
            }
            inferred_dim = i;
        } else {
            output_volume = output_volume * logical_shape[i];
        }
    }
    if (inferred_dim == -1) {
        return {logical_shape, padded_shape};
    }

    uint32_t implied_dim_value = (output_volume == 0) ? 0 : input_volume / output_volume;
    ttnn::Shape new_shape = logical_shape;
    new_shape[inferred_dim] = implied_dim_value;
    return {new_shape, new_shape};
}

ttnn::Tensor reshape_tiled(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const MemoryConfig& memory_config,
    const PadValue& /*pad_value*/,
    const bool recreate_mapping_tensor,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // squeeze input tensor and requested shape to 3D
    auto transform_to_3d = [](const auto& shape) -> ttnn::Shape {
        if (shape.rank() > 3) {
            return squeeze_shape_to_3D(shape);
        }
        if (shape.rank() < 3) {
            return unsqueeze_shape_to_3D(shape);
        }
        return shape;
    };

    const auto input_tensor_shape_3d = transform_to_3d(tensor.logical_shape());
    const auto requested_shape_3d = transform_to_3d(logical_shape);

    const auto requested_padded_shape_3d = compute_padded_shape(requested_shape_3d);
    const auto input_padded_shape_3d = compute_padded_shape(input_tensor_shape_3d);
    auto tensor3d = PerformView(tensor, input_tensor_shape_3d, input_padded_shape_3d);

    if (tensor.dtype() == DataType::BFLOAT8_B) {
        TT_FATAL(!sub_core_grid.has_value(), "Bfloat8 reshape does not support sub core grid specification\n");
        tensor3d = ttnn::typecast(tensor3d, DataType::BFLOAT16);
    }

    auto updated_mem_config = memory_config;
    if (updated_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto shard_spec = updated_mem_config.shard_spec().value();
        shard_spec.shape[1] = requested_shape_3d[-1];
        updated_mem_config = updated_mem_config.with_shard_spec(shard_spec);
    } else if (updated_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        auto shard_spec = updated_mem_config.shard_spec().value();
        shard_spec.shape[0] = requested_shape_3d[-2];
        updated_mem_config = updated_mem_config.with_shard_spec(shard_spec);
    }

    auto output_tensor_3d = ttnn::prim::reshape_view(
        tensor3d,
        requested_shape_3d,
        requested_padded_shape_3d,
        updated_mem_config,
        recreate_mapping_tensor,
        sub_core_grid);

    if (updated_mem_config.is_sharded()) {
        TT_FATAL(!sub_core_grid.has_value(), "Sharded reshape does not support sub core grid specification\n");

        // Recompute the shard spec for the output tensor shape
        auto output_mem_config =
            detail::recompute_shard_spec_for_output(updated_mem_config, output_tensor_3d.tensor_spec());
        output_tensor_3d = ttnn::prim::reshape_view(
            tensor3d,
            requested_shape_3d,
            requested_padded_shape_3d,
            output_mem_config,
            recreate_mapping_tensor,
            sub_core_grid);
    }

    if (tensor.dtype() == DataType::BFLOAT8_B) {
        TT_FATAL(!sub_core_grid.has_value(), "Bfloat8 reshape does not support sub core grid specification\n");
        output_tensor_3d = ttnn::typecast(output_tensor_3d, tensor.dtype());
    }

    return PerformView(output_tensor_3d, logical_shape, compute_padded_shape(logical_shape));
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_input_shape,
    const ttnn::Shape& padded_input_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    MemoryConfig mem_config = memory_config.value_or(tensor.memory_config());
    auto layout = tensor.layout();
    auto tensor_shape = tensor.logical_shape();

    const auto [logical_shape, padded_shape] = shape_corrector(tensor, logical_input_shape, padded_input_shape);
    // First Case, No reshape Required
    if (tensor.logical_shape() == logical_shape && tensor.padded_shape() == padded_shape) {
        return tensor;
    }
    PadValue default_pad_value;
    if (tensor.dtype() == DataType::BFLOAT8_B or tensor.dtype() == DataType::BFLOAT16 or
        tensor.dtype() == DataType::FLOAT32) {
        default_pad_value = 0.0f;
    } else {
        default_pad_value = (uint32_t)0;
    }

    const uint32_t tile_first_dim = tt::constants::TILE_HEIGHT;
    const uint32_t tile_second_dim = tt::constants::TILE_WIDTH;

    // The following case should only be called for the device storage case, the rest is a bandaid
    // for issue 15317

    const uint32_t shape_last_dim = logical_shape.rank() >= 1 ? logical_shape[-1] : 1;
    const uint32_t tensor_shape_last_dim = tensor_shape.rank() >= 1 ? tensor_shape[-1] : 1;
    const uint32_t shape_second_last_dim = logical_shape.rank() >= 2 ? logical_shape[-2] : 1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;

    // Just edit shape if shape has a 0 dimension
    if (tensor.logical_volume() == 0) {
        TT_FATAL(logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
        return ttnn::experimental::view(tensor, logical_shape, padded_shape);
    }
    TT_FATAL(logical_shape.volume() != 0, "Tensor volume is not 0, but shape volume is 0");

    if (!is_device_tensor(tensor)) {
        // This case has been allowed in the past though it means introducing padding values to the data
        return ttnn::experimental::view(tensor, logical_shape, padded_shape);
    }

    bool this_is_view =
        (tensor_shape_last_dim == shape_last_dim) && (mem_config.is_sharded() == tensor.memory_config().is_sharded()) &&
        (mem_config.is_l1() == tensor.memory_config().is_l1()) &&
        ((tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) ||              // Its row major
         (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
         (shape_second_last_dim % tile_second_dim == 0 &&
          tensor_shape_second_last_dim % tile_first_dim == 0));  // There is no padding on the second last dimension

    if (this_is_view) {
        return PerformView(tensor, logical_shape, padded_shape, tile_first_dim, tile_second_dim);
    }
    if (logical_shape.volume() != tensor.logical_volume()) {
        // This is completely incorrect but it is due to issue 15137 or issue 15558
        const auto& tile = tensor.tensor_spec().tile();
        bool tile_tensor_view_reshape_possible =
            (layout == ttnn::Layout::TILE and padded_shape.rank() >= 2 and padded_shape[-2] % tile.get_height() == 0 and
             padded_shape[-1] % tile.get_width() == 0 and tensor.padded_shape()[-1] == padded_shape[-1]);

        if (tile_tensor_view_reshape_possible) {
            // This case has been allowed in the past though it means introducing padding values to the data
            return ttnn::experimental::view(tensor, logical_shape, padded_shape);
        }
        // This is a completely incorrect test but it is due to issue 15558
        TT_FATAL(false, "Attempting to reshape between two shapes with different volumes");
    }
    // Do the reshape in row-major
    if (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return detail::reshape_rm(
            tensor,
            logical_shape,
            padded_shape,
            tile_first_dim,
            tile_second_dim,
            mem_config,
            pad_value.value_or(default_pad_value),
            sub_core_grid);
    }
    return reshape_tiled(
        tensor,
        logical_shape,
        mem_config,
        pad_value.value_or(default_pad_value),
        reshape_map_mode == TileReshapeMapMode::RECREATE,
        sub_core_grid);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return invoke(tensor, shape, shape, memory_config, pad_value, reshape_map_mode, sub_core_grid);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return invoke(
        tensor,
        detail::infer_dims_for_reshape(tensor, shape_vector),
        memory_config,
        pad_value,
        reshape_map_mode,
        sub_core_grid);
}

}  // namespace ttnn::operations::data_movement
