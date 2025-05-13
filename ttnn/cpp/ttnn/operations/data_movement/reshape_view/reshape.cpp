// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/constants.hpp>

#include "ttnn/common/queue_id.hpp"

#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "reshape.hpp"
#include "reshape_common.hpp"
#include "device/reshape_device_operation.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

ttnn::Tensor convert_tile_to_rm(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value) {
    // Convert the 3D->3D reshaping to row major and back to tile
    TT_FATAL(
        !(((logical_shape[-1] % tile_first_dim != 0) || (logical_shape[-2] % tile_second_dim != 0) ||
           (tensor.get_logical_shape()[-1] % tile_first_dim != 0) ||
           (tensor.get_logical_shape()[-2] % tile_second_dim != 0)) &&
          (tensor.get_dtype() == DataType::BFLOAT8_B)),
        "illegal dimensions for a bfloat8 tensor");
    auto new_tensor = (tensor.get_dtype() == DataType::BFLOAT8_B) ? ttnn::typecast(tensor, DataType::BFLOAT16) : tensor;
    new_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (IDevice*)nullptr);
    new_tensor =
        ReshapeViewOperation::invoke(queue_id, new_tensor, logical_shape, padded_shape, memory_config, pad_value);
    new_tensor =
        ttnn::to_layout(new_tensor, ttnn::TILE_LAYOUT, new_tensor.get_dtype(), memory_config, (IDevice*)nullptr);
    new_tensor =
        (tensor.get_dtype() == DataType::BFLOAT8_B) ? ttnn::typecast(new_tensor, tensor.get_dtype()) : new_tensor;
    return new_tensor;
}

ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const MemoryConfig& memory_config,
    const QueueId queue_id) {
    auto temp_tensor = tensor;
    auto intermediate_mem_config = tensor.memory_config();
    auto intermediate_out_memory_config = memory_config;
    if (tensor.memory_config().is_sharded()) {
        MemoryConfig temp_memory_config{TensorMemoryLayout::INTERLEAVED, tensor.memory_config().buffer_type()};
        temp_tensor = ttnn::sharded_to_interleaved(queue_id, tensor, temp_memory_config, std::nullopt);
    }
    if (memory_config.is_sharded()) {
        intermediate_out_memory_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, intermediate_out_memory_config.buffer_type()};
    }
    // Guaranteed to be interleaved
    // We are guaranteed to be working 2D->2D in this function
    auto temp_tensor2 = tt::tt_metal::operation::run(
                            ReshapeDeviceOperation{logical_shape, padded_shape, intermediate_out_memory_config},
                            {temp_tensor},
                            {},
                            {},
                            queue_id)
                            .at(0);
    if (memory_config.is_sharded()) {
        return ttnn::interleaved_to_sharded(queue_id, temp_tensor2, memory_config, std::nullopt);
    } else {
        return temp_tensor2;
    }
}

ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id) {
    //This function turns a RM 2D->MD into an equivalent 2D->2D conversion and then turns the 2D output back to MD using a 0 cost view
    TT_FATAL((logical_shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    //Collapse into the second last dimension
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
            queue_id),
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
    const QueueId queue_id,
    const PadValue& pad_value) {
    // This function turns ND -> MD into 2D->MD for row major and 3D->MD for tiled using a 0 cost view
    const auto& tensor_shape = tensor.get_logical_shape();
    TT_FATAL((tensor_shape.rank() != 0), "Can't do reshape from rank 0 tensor");
    TT_FATAL(tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT, "Wrong layout in `reshape_rm` `");

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
        queue_id);
}
}  // namespace detail

ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim = tt::constants::TILE_HEIGHT,
    const uint32_t tile_second_dim = tt::constants::TILE_WIDTH) {
    if (tensor.get_logical_shape() == logical_shape && tensor.get_padded_shape() == padded_shape) {
        return tensor;
    }
    if (logical_shape.rank() == 1) {
        return ttnn::experimental::view(tensor, logical_shape);
    } else if (
        tensor.get_layout() == ttnn::TILE_LAYOUT &&
        (logical_shape[-1] % tile_first_dim != 0 || logical_shape[-2] % tile_second_dim != 0)) {
        return ttnn::experimental::view(tensor, logical_shape, compute_padded_shape(logical_shape));
    }
    //Perform a reshape (view)
    return ttnn::experimental::view(tensor, logical_shape, padded_shape);
}

std::pair<ttnn::Shape, ttnn::Shape> shape_corrector(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    //Correct the shape to account for inferred dimensions
    uint32_t input_volume = tensor.get_logical_volume();
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
    if (inferred_dim == -1)
    {
        return {logical_shape, padded_shape};
    }

    uint32_t implied_dim_value = (output_volume == 0) ? 0: input_volume/output_volume;
    ttnn::Shape new_shape = logical_shape;
    new_shape[inferred_dim] = implied_dim_value;
    return {new_shape, new_shape};
}

ttnn::Tensor reshape_tiled(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value) {
    // squeeze input tensor and requested shape to 3D

    auto transform_to_3d = [](const auto& shape) -> ttnn::Shape {
        if (shape.rank() > 3) {
            return squeeze_shape_to_3D(shape);
        } else if (shape.rank() < 3) {
            return unsqueeze_shape_to_3D(shape);
        } else {
            return shape;
        }
    };

    const auto input_tensor_shape_3d = transform_to_3d(tensor.get_logical_shape());
    const auto requested_shape_3d = transform_to_3d(logical_shape);

    const auto requested_padded_shape_3d = compute_padded_shape(requested_shape_3d);
    const auto input_padded_shape_3d = compute_padded_shape(input_tensor_shape_3d);
    auto tensor3d = PerformView(tensor, input_tensor_shape_3d, input_padded_shape_3d);

    if (tensor.memory_config().is_sharded()) {
        MemoryConfig working_input_memory_config{TensorMemoryLayout::INTERLEAVED, tensor.memory_config().buffer_type()};
        tensor3d = ttnn::sharded_to_interleaved(queue_id, tensor, working_input_memory_config, std::nullopt);
    }

    if (tensor.get_dtype() == DataType::BFLOAT8_B) {
        tensor3d = ttnn::typecast(tensor3d, DataType::BFLOAT16);
    }

    MemoryConfig working_output_memory_config = memory_config;
    if (memory_config.is_sharded()) {
        working_output_memory_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, working_output_memory_config.buffer_type()};
    }

    auto output_tensor_3d =
        tt::tt_metal::operation::run(
            ReshapeDeviceOperation{requested_shape_3d, requested_padded_shape_3d, working_output_memory_config},
            {tensor3d},
            {},
            {},
            queue_id)
            .at(0);

    if (memory_config.is_sharded()) {
        output_tensor_3d = ttnn::interleaved_to_sharded(queue_id, output_tensor_3d, memory_config, std::nullopt);
    }

    if (tensor.get_dtype() == DataType::BFLOAT8_B) {
        output_tensor_3d = ttnn::typecast(output_tensor_3d, tensor.get_dtype());
    }

    return PerformView(output_tensor_3d, logical_shape, compute_padded_shape(logical_shape));
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const QueueId queue_id,
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_input_shape,
    const ttnn::Shape& padded_input_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value) {
    MemoryConfig mem_config = memory_config.value_or(tensor.memory_config());
    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_logical_shape();

    const auto [logical_shape, padded_shape] = shape_corrector(tensor, logical_input_shape, padded_input_shape);
    // First Case, No reshape Required
    if (tensor.get_logical_shape() == logical_shape && tensor.get_padded_shape() == padded_shape) {
        return tensor;
    }
    PadValue default_pad_value;
    if (tensor.get_dtype() == DataType::BFLOAT8_B or tensor.get_dtype() == DataType::BFLOAT16 or
        tensor.get_dtype() == DataType::FLOAT32) {
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
    if (tensor.get_logical_volume() == 0) {
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
        ((tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) ||          // Its row major
         (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
         (shape_second_last_dim % tile_second_dim == 0 &&
          tensor_shape_second_last_dim % tile_first_dim == 0));  // There is no padding on the second last dimension

    if (this_is_view) {
        return PerformView(tensor, logical_shape, padded_shape, tile_first_dim, tile_second_dim);
    }
    if (logical_shape.volume() != tensor.get_logical_volume()) {
        // This is completely incorrect but it is due to issue 15137 or issue 15558
        const auto& tile = tensor.tensor_spec().tile();
        bool tile_tensor_view_reshape_possible =
            (layout == ttnn::Layout::TILE and padded_shape.rank() >= 2 and padded_shape[-2] % tile.get_height() == 0 and
             padded_shape[-1] % tile.get_width() == 0 and tensor.get_padded_shape()[-1] == padded_shape[-1]);

        if (tile_tensor_view_reshape_possible) {
            // This case has been allowed in the past though it means introducing padding values to the data
            return ttnn::experimental::view(tensor, logical_shape, padded_shape);
        }
        // This is a completely incorrect test but it is due to issue 15558
        TT_FATAL(false, "Attempting to reshape between two shapes with different volumes");
    }
    // Do the reshape in row-major
    if (tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return detail::reshape_rm(
            tensor,
            logical_shape,
            padded_shape,
            tile_first_dim,
            tile_second_dim,
            mem_config,
            queue_id,
            pad_value.value_or(default_pad_value));
    } else {
        return reshape_tiled(tensor, logical_shape, mem_config, queue_id, pad_value.value_or(default_pad_value));
    }
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const QueueId queue_id,
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value) {
    return invoke(queue_id, tensor, shape, shape, memory_config, pad_value);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const QueueId queue_id,
    const ttnn::Tensor& tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value) {
    return invoke(
        queue_id, tensor, tt::tt_metal::infer_dims_for_reshape(tensor, shape_vector), memory_config, pad_value);
}

}  // namespace ttnn::operations::data_movement
