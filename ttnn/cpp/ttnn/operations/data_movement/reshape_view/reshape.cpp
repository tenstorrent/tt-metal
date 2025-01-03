// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "reshape_common.hpp"
#include "tt_metal/common/constants.hpp"
#include <functional>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/reshape_rm_op.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

namespace ttnn::operations::data_movement {


namespace detail {

ttnn::Tensor convert_tile_to_rm(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig &memory_config,
    const uint8_t queue_id,
    const PadValue &pad_value
) {
    // Convert the 3D->3D reshaping to row major and back to tile
    TT_FATAL(
        !(((shape[-1] % tile_first_dim != 0) || (shape[-2] % tile_second_dim != 0) ||
           (tensor.get_shape()[-1] % tile_first_dim != 0) || (tensor.get_shape()[-2] % tile_second_dim != 0)) &&
          (tensor.get_dtype() == DataType::BFLOAT8_B)),
        "illegal dimensions for a bfloat8 tensor");
    auto new_tensor = (tensor.get_dtype() == DataType::BFLOAT8_B) ? ttnn::typecast(tensor, DataType::BFLOAT16) : tensor;
    new_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    new_tensor = ReshapeViewOperation::invoke(new_tensor, shape, memory_config, queue_id, pad_value);
    new_tensor =
        ttnn::to_layout(new_tensor, ttnn::TILE_LAYOUT, new_tensor.get_dtype(), memory_config, (Device*)nullptr);
    new_tensor =
        (tensor.get_dtype() == DataType::BFLOAT8_B) ? ttnn::typecast(new_tensor, tensor.get_dtype()) : new_tensor;
    return new_tensor;
}

//Wrapper to turn the ND-> MD problem into 3D->3D for tiled and 2D->2D for Row Major

ttnn::Tensor convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig &memory_config,
    const uint8_t queue_id,
    const PadValue &pad_value
    )
{
    //This function turns ND -> MD into 2D->MD for row major and 3D->MD for tiled using a 0 cost view
    const auto layout = tensor.get_layout();
    const auto tensor_shape = tensor.get_shape();
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
            PerformView
            (
                tensor,
                ttnn::Shape{second_dim,tensor_shape[-1]},
                tile_first_dim,
                tile_second_dim
            ),
            shape,
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id
        );
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
            PerformView
            (
                tensor,
                ttnn::Shape{third_dim,second_dim,tensor_shape[-1]},
                tile_first_dim,
                tile_second_dim
            )
            ,shape,
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id,
            pad_value
        );
    }
    TT_FATAL(false, "Layout is neither tile nor row major");

}

ttnn::Tensor fix_shape_and_perform_reshape_on_3D_TILE(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig &memory_config,
    const uint8_t queue_id,
    const PadValue &pad_value
    )
{
    //This function turns a TILE 3D->MD into an equivalent 3D->3D conversion and then turns the 3D output back to MD using a 0 cost view
    //Collapse into the third last dimension
    TT_FATAL((shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    uint32_t third_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.rank()) - 2; ++i)
    {
        third_dim = third_dim * shape[i];
    }
    //Figure out the second last dimension
    const uint32_t second_dim = shape.rank() > 1 ? shape[-2] : 1;
    return PerformView
    (
        convert_tile_to_rm(
            tensor,
            ttnn::Shape{third_dim,second_dim,shape[-1]},
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id,
            pad_value
        ),
        shape,
        tile_first_dim,
        tile_second_dim);
}

ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig &memory_config,
    const uint8_t queue_id
    )
{
    //This function turns a RM 2D->MD into an equivalent 2D->2D conversion and then turns the 2D output back to MD using a 0 cost view
    TT_FATAL((shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    //Collapse into the second last dimension
    uint32_t second_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.rank()) - 1; ++i)
    {
        second_dim = second_dim * shape[i];
    }
    return PerformView
    (
        perform_reshape_on_2D_RM
        (
            tensor,
            ttnn::Shape{second_dim,shape[-1]},
            memory_config,
            queue_id
        ),
        shape,
        tile_first_dim,
        tile_second_dim
    );
}

//Entry points into device prep code

ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const MemoryConfig &memory_config,
    const uint8_t queue_id
    )
{
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
    auto temp_tensor2 =  operation::run(
        RM_RESHAPE_STRUCT
        {
            shape,
            intermediate_out_memory_config
        },
        {temp_tensor},
        {},
        {},
        queue_id
    ).at(0);
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

ttnn::Shape tiling_reshape_corrector(const ttnn::Shape& shape, const uint32_t tile_first_dim, const uint32_t tile_second_dim) {
    //Apply the correct padding metadata to the target shape
    ttnn::Shape padded = shape.with_tile_padding();
    int64_t rank = shape.rank();
    const int8_t correction_1 =(tile_first_dim - (int)padded[-1] % tile_first_dim) % tile_first_dim;
    if(rank == 1)
    {
        return ttnn::Shape({1, shape[0]}, {32, padded[0] + correction_1});
    }
    const int8_t correction_2 =(tile_second_dim - (int)padded[-2] % tile_second_dim) % tile_second_dim;
    switch(rank)
    {
        case 2:
            return ttnn::Shape({shape[0],shape[1]},{padded[0]+correction_2,padded[1]+correction_1});
            break;
        case 3:
            return ttnn::Shape({shape[0],shape[1],shape[2]},{padded[0],padded[1]+correction_2,padded[2]+correction_1});
            break;
        case 4:
            return ttnn::Shape({shape[0],shape[1],shape[2],shape[3]},{padded[0],padded[1],padded[2]+correction_2,padded[3]+correction_1});
            break;
        case 5:
            return ttnn::Shape(
                {shape[0], shape[1], shape[2], shape[3], shape[4]},
                {padded[0], padded[1], padded[2], padded[3] + correction_2, padded[4] + correction_1});
            break;
        case 6:
            return ttnn::Shape(
                {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]},
                {padded[0], padded[1], padded[2], padded[3], padded[4] + correction_2, padded[5] + correction_1});
            break;
        case 7:
            return ttnn::Shape(
                {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]},
                {padded[0],
                 padded[1],
                 padded[2],
                 padded[3],
                 padded[4],
                 padded[5] + correction_2,
                 padded[6] + correction_1});
            break;
        case 8:
            return ttnn::Shape(
                {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]},
                {padded[0],
                 padded[1],
                 padded[2],
                 padded[3],
                 padded[4],
                 padded[5],
                 padded[6] + correction_2,
                 padded[7] + correction_1});
            break;
    }
    return shape;
}

ttnn::Tensor PerformView(const ttnn::Tensor& tensor, const ttnn::Shape& shape, const uint32_t tile_first_dim, const uint32_t tile_second_dim) {
    if (tensor.get_shape() == shape) {
        return tensor;
    }
    if (tensor.get_layout() == ttnn::TILE_LAYOUT &&
        (shape[-1]%tile_first_dim!=0 || shape.rank()==1 || shape[-2]%tile_second_dim!=0 ))
    {
        //Correct the output shape to add padding metadata before reshape (view)
        return tensor.reshape(tiling_reshape_corrector(shape, tile_first_dim, tile_second_dim));
    }
    //Perform a reshape (view)
    return tensor.reshape(shape);
}

ttnn::Shape shape_corrector(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    //Correct the shape to account for inferred dimensions
    uint32_t input_volume = tensor.get_logical_volume();
    uint32_t output_volume = 1;
    uint32_t inferred_dim = -1;
    for (uint32_t i=0; i< shape.rank(); i++) {
        if ((static_cast<int>(shape[i])) == -1) {
            if (inferred_dim != -1) {
                TT_FATAL(false, "Only one dimension can be inferred in reshape");
            }
            inferred_dim = i;
        } else {
            output_volume = output_volume * shape[i];
        }
    }
    if (inferred_dim == -1)
    {
        return shape;
    }

    uint32_t implied_dim_value = (output_volume == 0) ? 0: input_volume/output_volume;
    ttnn::SmallVector<uint32_t> new_shape(shape.size());
    auto old_shape = shape.logical_shape().view();
    std::copy(old_shape.begin(), old_shape.end(), new_shape.begin());
    new_shape[inferred_dim] = implied_dim_value;
    return ttnn::Shape(std::move(new_shape));
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& input_shape,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id,
    const std::optional<PadValue>& pad_value) {
    MemoryConfig mem_config = memory_config.value_or(tensor.memory_config());
    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_shape();
    const ttnn::Shape shape = shape_corrector(tensor, input_shape);
    // First Case, No reshape Required
    if (tensor_shape == shape) {
        return tensor;
    }
    PadValue default_pad_value;
    if (tensor.get_dtype() == DataType::BFLOAT8_B or tensor.get_dtype() == DataType::BFLOAT16 or
        tensor.get_dtype() == DataType::FLOAT32) {
        default_pad_value = 0.0f;
    } else {
        default_pad_value = (uint32_t)0;
    }

    //const uint32_t tile_first_dim =tensor.get_tile().get_width();
    //const uint32_t tile_second_dim =tensor.get_tile().get_height();
    const uint32_t tile_first_dim = 32;
    const uint32_t tile_second_dim = 32;
    //The following case should only be called for the device storage case, the rest is a bandaid
    //for issue 15317

    const uint32_t shape_second_last_dim = shape.rank() >= 2 ? shape[-2]:1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2]:1;

    // Just edit shape if shape has a 0 dimension
    if (tensor.get_logical_volume() == 0) {
        TT_FATAL(shape.logical_shape().volume() == 0, "Tensor volume is 0, but shape's volume is not");
        TT_FATAL(
            (tensor.storage_type() != StorageType::MULTI_DEVICE &&
             tensor.storage_type() != StorageType::MULTI_DEVICE_HOST),
            "Reshaping a multi-device tensor with 0 volume is not supported");
        return tensor.reshape(shape);
    }
    TT_FATAL(shape.logical_shape().volume() != 0, "Tensor volume is not 0, but shape volume is 0");

    bool this_is_view =
        (tensor_shape[-1] == shape[-1]) && (mem_config.is_sharded() == tensor.memory_config().is_sharded()) &&
        (mem_config.is_l1() == tensor.memory_config().is_l1()) &&
        ((tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) ||          // Its row major
         (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
         (shape_second_last_dim % tile_second_dim == 0 &&
          tensor_shape_second_last_dim % tile_first_dim == 0));  // There is no padding on the second last dimension
    if (!(ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE))) {
            // This case has been allowed in the past though it means introducing padding values to the data
            return tensor.reshape(shape);
        }

    if (this_is_view) {
        return PerformView(tensor,shape, tile_first_dim, tile_second_dim);
    }
    if (shape.logical_shape().volume() != tensor.get_logical_volume()) {
        // This is completely incorrect but it is due to issue 15137 or issue 15558
        const auto& tile = tensor.tensor_spec().tile();
        bool tile_tensor_view_reshape_possible =
            (layout == ttnn::Layout::TILE and shape.with_tile_padding().rank() >= 2 and
             shape.with_tile_padding()[-2] % tile.get_height() == 0 and
             shape.with_tile_padding()[-1] % tile.get_width() == 0 and
             tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]);

        if (tile_tensor_view_reshape_possible) {
            // This case has been allowed in the past though it means introducing padding values to the data
            return tensor.reshape(shape);
        }
        // This is a completely incorrect test but it is due to issue 15558
        TT_FATAL(false, "Attempting to reshape between two shapes with different volumes");
    }
    // Catch-all
    // Do the reshape in row-major
    return detail::convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
        tensor,
        shape,
        tile_first_dim,
        tile_second_dim,
        mem_config,
        queue_id,
        pad_value.value_or(default_pad_value)
        );
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape
     ) {
        return invoke(tensor, shape,std::nullopt,0,std::nullopt);
     }

     ttnn::Tensor ReshapeViewOperation::invoke(
         const ttnn::Tensor& tensor,
         const ttnn::SimpleShape& shape,
         const std::optional<MemoryConfig>& memory_config,
         const uint8_t queue_id,
         const std::optional<PadValue>& pad_value) {
         return invoke(tensor, ttnn::Shape(shape.view()), memory_config, queue_id, pad_value);
     }

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::SimpleShape& shape
    ) {
    return invoke(tensor, ttnn::Shape(shape.view()), std::nullopt, 0, std::nullopt);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig> &memory_config,
    const uint8_t queue_id,
    const std::optional<PadValue> &pad_value
    ) {
    return invoke(tensor, tt::tt_metal::infer_dims_for_reshape(tensor, shape_vector),memory_config,queue_id,pad_value);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    tt::stl::Span<const int32_t> shape_vector
    ) {
    return invoke(tensor, tt::tt_metal::infer_dims_for_reshape(tensor, shape_vector),std::nullopt,0,std::nullopt);
}

} // ttnn::operations::data_movement namespace
