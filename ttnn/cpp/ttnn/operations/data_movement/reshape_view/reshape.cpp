// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "reshape_common.hpp"
#include "tt_metal/common/constants.hpp"
#include <functional>
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/reshape_rm_op.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

namespace ttnn::operations::data_movement {


namespace detail {

ttnn::Tensor host_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    //This function is due to embedding issue 15558, once the issue is fixed we want to delete it
    tt::log_warning("host_reshape is deprecated and will be removed in the near future");
    if (!ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
        return tensor.reshape(shape);
    }
    auto tensor_shape = tensor.shape();
    auto layout = tensor.layout();
    auto device = tensor.device();
    auto memory_config = tensor.memory_config();
    auto host_tensor = tensor.cpu();
    auto rm_tensor = ttnn::to_layout(host_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);

    if (tensor_shape.has_tile_padding()) {
        ttnn::Tensor slice_input;
        auto host_tensor_4d = unsqueeze_to_4D(rm_tensor);
        auto tensor_shape_4d = host_tensor_4d.shape();
        ttnn::SmallVector<uint32_t> begins({0, 0, 0, 0});
        ttnn::SmallVector<uint32_t> ends(
            {tensor_shape_4d[0], tensor_shape_4d[1], tensor_shape_4d[2], tensor_shape_4d[3]});
        ttnn::SmallVector<uint32_t> step({1, 1, 1, 1});
        host_tensor_4d = ttnn::slice(host_tensor_4d, begins, ends, step, std::nullopt);
        host_tensor = squeeze_from_4D(host_tensor_4d, tensor_shape.rank());
    }
    auto host_reshape_tensor = rm_tensor.reshape(shape);
    auto final_layout_tensor =
        ttnn::to_layout(host_reshape_tensor, layout, std::nullopt, std::nullopt, (Device*)nullptr);
    auto device_tensor = ttnn::data_transfer_to_device(final_layout_tensor, device, memory_config);
    return device_tensor;
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
    TT_FATAL((tensor_shape.rank()!=0), "can't do reshape from rank 0 tensor");
    if(layout == ttnn::ROW_MAJOR_LAYOUT)
    {
        //Collapse into the second last dimension
        uint32_t second_dim = 1;
        for (int i=0; i <tensor_shape.rank()-1; i++)
        {
            second_dim = second_dim * tensor_shape[i];
        }
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
        auto rm_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        rm_tensor = convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
            rm_tensor,
            shape,
            tile_first_dim,
            tile_second_dim,
            memory_config,
            queue_id,
            pad_value
        );
        rm_tensor = ttnn::to_layout(rm_tensor, ttnn::Layout::TILE, std::nullopt, std::nullopt, (Device*)nullptr);
        return rm_tensor;
    }
    TT_FATAL(false, "layout is neither tile nor row major");

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
    TT_FATAL((shape.rank()!=0), "can't do reshape to rank 0 tensor");
    //Collapse into the second last dimension
    uint32_t second_dim = 1;
    for (int i=0; i <shape.rank()-1; i++)
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
    auto padded = shape.with_tile_padding();
    auto rank = shape.rank();
    const int8_t correction_1 =(tile_first_dim - (int)padded[-1] % tile_first_dim) % tile_first_dim;
    if(rank == 1)
    {
        return ttnn::Shape({1,shape[0]},{32,padded[0]+correction_1});
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

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig> &memory_config,
    const uint8_t queue_id,
    const std::optional<PadValue> &pad_value
     ) {
    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_shape();

    // First Case, No reshape Required
    if (tensor_shape == shape) {
        return tensor;
    }
    PadValue default_pad_value;
    if(tensor.get_dtype() == DataType::BFLOAT16 or tensor.get_dtype() == DataType::FLOAT32) {
        default_pad_value = 0.0f;
    }
    else {
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
    bool this_is_view = (tensor_shape[-1] == shape[-1]) &&
        ((tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) || //Its row major
        (tensor_shape_second_last_dim==shape_second_last_dim) || //Second last dimension is the same
        (shape_second_last_dim % tile_second_dim==0 && tensor_shape_second_last_dim % tile_first_dim==0)); //There is no padding on the second last dimension
    bool tile_tensor_view_reshape_possible =
        (layout == ttnn::Layout::TILE and shape.with_tile_padding().rank() >= 2 and
         shape.with_tile_padding()[-2] % ttnn::TILE_SIZE == 0 and
         shape.with_tile_padding()[-1] % ttnn::TILE_SIZE == 0 and
         tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]);

    if (!(ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) or tile_tensor_view_reshape_possible) {
        // This case has been allowed in the past though it means introducing padding values to the data
        return tensor.reshape(shape);
    }


    if (this_is_view) {
        return PerformView(tensor,shape, tile_first_dim, tile_second_dim);
    }
    if(shape.logical_shape().volume() != tensor.get_logical_volume())
    {
        //This is a completely incorrect test but it is due to issue 15558
        return detail::host_reshape(tensor, shape);
    }
    // Catch-all
    // Do the reshape in row-major
    return detail::convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
        tensor,
        shape,
        tile_first_dim,
        tile_second_dim,
        memory_config.value_or(tensor.memory_config()),
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
    const std::optional<MemoryConfig> &memory_config,
    const uint8_t queue_id,
    const std::optional<PadValue> &pad_value
    ) {
    return invoke(tensor, ttnn::Shape(shape.view()),memory_config,queue_id,pad_value);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::SimpleShape& shape
    ) {
    return invoke(tensor, ttnn::Shape(shape.view()),std::nullopt,0,std::nullopt);
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
