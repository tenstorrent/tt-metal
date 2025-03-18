// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "reshape_common.hpp"
#include <tt-metalium/constants.hpp>
#include <functional>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/reshape_rm_op.hpp"
#include "cpp/ttnn/operations/copy.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

#include "cpp/ttnn/operations/experimental/reshape/view.hpp"

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

//Wrapper to turn the ND-> MD problem into 3D->3D for tiled and 2D->2D for Row Major

ttnn::Tensor convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value) {
    //This function turns ND -> MD into 2D->MD for row major and 3D->MD for tiled using a 0 cost view
    const auto layout = tensor.get_layout();
    const auto& tensor_shape = tensor.get_logical_shape();
    TT_FATAL((tensor_shape.rank() != 0), "Can't do reshape from rank 0 tensor");
    if(layout == ttnn::ROW_MAJOR_LAYOUT)
    {
        //Collapse into the second last dimension
        uint32_t second_dim = 1;
        for (int64_t i = 0; i < static_cast<int64_t>(tensor_shape.rank()) - 1; ++i)
        {
            second_dim = second_dim * tensor_shape[i];
        }
        //Call reshape with the equivalent data 2D Row Major input tensor
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
    else if (layout == ttnn::Layout::TILE)
    {
        uint32_t third_dim = 1;
        //Collapse into the third last dimension
        for (int64_t i = 0; i < static_cast<int64_t>(tensor_shape.rank()) - 2; ++i)
        {
            third_dim = third_dim * tensor_shape[i];
        }
        //Figure out the second last dimension
        const uint32_t second_dim = tensor_shape.rank() > 1 ? tensor_shape[-2] : 1;
        //Call reshape with the equivalent data 3D Tile input tensor
        return fix_shape_and_perform_reshape_on_3D_TILE(
            PerformView(
                tensor,
                Shape({third_dim, second_dim, tensor_shape[-1]}),
                Shape({third_dim, second_dim, tensor_shape[-1]}),
                tile_first_dim,
                tile_second_dim),
            logical_shape,
            padded_shape,
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id,
            pad_value);
    }
    TT_FATAL(false, "Layout is neither tile nor row major");
}

ttnn::Tensor fix_shape_and_perform_reshape_on_3D_TILE(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value) {
    //This function turns a TILE 3D->MD into an equivalent 3D->3D conversion and then turns the 3D output back to MD using a 0 cost view
    //Collapse into the third last dimension
    TT_FATAL((logical_shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    uint32_t third_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(logical_shape.rank()) - 2; ++i) {
        third_dim = third_dim * logical_shape[i];
    }
    //Figure out the second last dimension
    const uint32_t second_dim = logical_shape.rank() > 1 ? logical_shape[-2] : 1;
    return PerformView(
        convert_tile_to_rm(
            tensor,
            ttnn::Shape{third_dim, second_dim, logical_shape[-1]},
            ttnn::Shape{third_dim, second_dim, logical_shape[-1]},
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id,
            pad_value),
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim);
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

//Entry points into device prep code

ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const MemoryConfig& memory_config,
    const QueueId queue_id) {
    auto temp_tensor = tensor;
    auto intermediate_mem_config = tensor.memory_config();
    auto intermediate_out_memory_config = memory_config;
    if(tensor.memory_config().is_sharded())
    {
        auto temp_memory_config = tensor.memory_config();
        temp_memory_config.memory_layout = TensorMemoryLayout::INTERLEAVED;
        temp_tensor = ttnn::sharded_to_interleaved(queue_id, tensor, temp_memory_config, std::nullopt);
    }
    if (memory_config.is_sharded())
    {
        intermediate_out_memory_config.memory_layout = TensorMemoryLayout::INTERLEAVED;
    }
    //Guaranteed to be interleaved
    //We are guaranteed to be working 2D->2D in this function
    auto temp_tensor2 = tt::tt_metal::operation::run(
                            RM_RESHAPE_STRUCT{logical_shape, padded_shape, intermediate_out_memory_config},
                            {temp_tensor},
                            {},
                            {},
                            queue_id)
                            .at(0);
    if(memory_config.is_sharded())
    {
        return ttnn::interleaved_to_sharded(queue_id,temp_tensor2, memory_config,std::nullopt);
    }
    else
    {
        return temp_tensor2;
    }
}
}

std::pair<ttnn::Shape, ttnn::Shape> tiling_reshape_corrector(
    const ttnn::Shape& shape,
    const ttnn::Shape& padded,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim) {
    // Apply the correct padding metadata to the target shape
    int64_t rank = shape.rank();
    const int8_t correction_1 =(tile_first_dim - (int)padded[-1] % tile_first_dim) % tile_first_dim;
    if(rank == 1)
    {
        return {ttnn::Shape({1, shape[0]}), ttnn::Shape({32, padded[0] + correction_1})};
    }
    const int8_t correction_2 =(tile_second_dim - (int)padded[-2] % tile_second_dim) % tile_second_dim;
    switch(rank)
    {
        case 2:
            return {
                ttnn::Shape({shape[0], shape[1]}), ttnn::Shape({padded[0] + correction_2, padded[1] + correction_1})};
            break;
        case 3:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2]}),
                ttnn::Shape({padded[0], padded[1] + correction_2, padded[2] + correction_1})};
            break;
        case 4:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2], shape[3]}),
                ttnn::Shape({padded[0], padded[1], padded[2] + correction_2, padded[3] + correction_1})};
            break;
        case 5:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2], shape[3], shape[4]}),
                ttnn::Shape({padded[0], padded[1], padded[2], padded[3] + correction_2, padded[4] + correction_1})};
            break;
        case 6:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]}),
                ttnn::Shape(
                    {padded[0], padded[1], padded[2], padded[3], padded[4] + correction_2, padded[5] + correction_1})};
            break;
        case 7:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]}),
                ttnn::Shape(
                    {padded[0],
                     padded[1],
                     padded[2],
                     padded[3],
                     padded[4],
                     padded[5] + correction_2,
                     padded[6] + correction_1})};
            break;
        case 8:
            return {
                ttnn::Shape({shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]}),
                ttnn::Shape(
                    {padded[0],
                     padded[1],
                     padded[2],
                     padded[3],
                     padded[4],
                     padded[5],
                     padded[6] + correction_2,
                     padded[7] + correction_1})};
            break;
    }
    return {shape, padded};
}

ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim) {
    if (tensor.get_logical_shape() == logical_shape && tensor.get_padded_shape() == padded_shape) {
        return tensor;
    }
    if (logical_shape.rank() == 1) {
        return ttnn::experimental::view(tensor, logical_shape);
    } else if (
        tensor.get_layout() == ttnn::TILE_LAYOUT &&
        (logical_shape[-1] % tile_first_dim != 0 || logical_shape[-2] % tile_second_dim != 0)) {
        //Correct the output shape to add padding metadata before reshape (view)
        auto [logical, padded] = tiling_reshape_corrector(logical_shape, padded_shape, tile_first_dim, tile_second_dim);
        return ttnn::experimental::view(tensor, logical, padded);
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

    // const uint32_t tile_first_dim =tensor.get_tile().get_width();
    // const uint32_t tile_second_dim =tensor.get_tile().get_height();
    const uint32_t tile_first_dim = 32;
    const uint32_t tile_second_dim = 32;
    // The following case should only be called for the device storage case, the rest is a bandaid
    // for issue 15317

    const uint32_t shape_last_dim = logical_shape.rank() >= 1 ? logical_shape[-1] : 1;
    const uint32_t tensor_shape_last_dim = tensor_shape.rank() >= 1 ? tensor_shape[-1] : 1;
    const uint32_t shape_second_last_dim = logical_shape.rank() >= 2 ? logical_shape[-2] : 1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;

    // Just edit shape if shape has a 0 dimension
    if (tensor.get_logical_volume() == 0) {
        TT_FATAL(logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
        TT_FATAL(
            (tensor.storage_type() != StorageType::MULTI_DEVICE &&
             tensor.storage_type() != StorageType::MULTI_DEVICE_HOST),
            "Reshaping a multi-device tensor with 0 volume is not supported");
        return ttnn::experimental::view(tensor, logical_shape, padded_shape);
    }
    TT_FATAL(logical_shape.volume() != 0, "Tensor volume is not 0, but shape volume is 0");

    if (!is_tensor_on_device_or_multidevice(tensor)) {
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
    // Catch-all
    // Do the reshape in row-major
    return detail::convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
        tensor,
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim,
        mem_config,
        queue_id,
        pad_value.value_or(default_pad_value));
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

} // ttnn::operations::data_movement namespace
