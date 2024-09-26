// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {


namespace detail {


    ttnn::Tensor host_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            auto tensor_shape = tensor.shape();
            auto layout = tensor.layout();
            auto device = tensor.device();
            auto memory_config = tensor.memory_config();
            auto host_tensor = tensor.cpu();
            auto rm_tensor = ttnn::to_layout(host_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
            auto host_tensor_4d = unsqueeze_to_4D(rm_tensor);
            auto tensor_shape_4d = host_tensor_4d.shape();
            if(tensor_shape.has_tile_padding()) {
                auto begins = std::vector<uint32_t>({0, 0, 0, 0});
                auto ends = std::vector<uint32_t>({tensor_shape_4d[0], tensor_shape_4d[1], tensor_shape_4d[2], tensor_shape_4d[3]});
                auto step = std::vector<uint32_t>({1, 1, 1, 1});
                host_tensor_4d = ttnn::slice(host_tensor_4d, begins, ends, step, std::nullopt);
                host_tensor = squeeze_from_4D(host_tensor_4d, tensor_shape.rank());
            }
            auto host_reshape_tensor = host_tensor.reshape(shape.value);
            auto final_layout_tensor = ttnn::to_layout(host_reshape_tensor, layout, std::nullopt, std::nullopt, (Device *)nullptr);
            auto device_tensor = ttnn::data_transfer_to_device(final_layout_tensor, device, memory_config);
            return device_tensor;
        }
        else{
            return tensor.reshape(shape.value);
        }

    }

    ttnn::Tensor row_major_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {

            const auto layout = tensor.get_layout();
            auto tensor_shape = tensor.get_shape();
            if (!ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
                return tensor.reshape(shape.value);
            }


            auto rm_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
            ttnn::Tensor reshaped_rm_tensor;
            if (rm_tensor.is_contiguous()) {
                // Page size depends on the width, so only modify the shape if the width is the same
                if (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]) {
                    return rm_tensor.reshape(shape.value);
                }
                //Different page width, need to remake tensor
                else {
                    uint32_t ROW_MAJOR_WIDTH = 8;
                    auto original_rank = shape.rank();

                    if(!(tensor_shape[-1] % ROW_MAJOR_WIDTH == 0 && shape[-1] % ROW_MAJOR_WIDTH == 0)) {
                        return  host_reshape(rm_tensor, shape);
                    }
                    else {
                        auto tensor_4d = unsqueeze_to_4D(rm_tensor);
                        const auto shape_4d = shape.to_rank<4>();
                        auto reshaped_tensor = ttnn::reshape_on_device(tensor_4d, shape_4d[0], shape_4d[1], shape_4d[2], shape_4d[3], tensor.memory_config());
                        reshaped_rm_tensor = squeeze_from_4D(reshaped_tensor, original_rank);
                    }
                }
            } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
                // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
                // is possible
                if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                    tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                    tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                    reshaped_rm_tensor = rm_tensor.reshape(shape.value);
                }
            }
            else {
                reshaped_rm_tensor = host_reshape(tensor, shape);
            }
            return ttnn::to_layout(reshaped_rm_tensor, layout, std::nullopt, std::nullopt, (Device *)nullptr);
    }


}


ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape
    ) {

    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_shape();
    if (tensor_shape == shape) {
        return tensor;
    }

    if (layout == ttnn::Layout::ROW_MAJOR) {
        return detail::row_major_reshape(tensor, shape);
    }
    else {
        const auto new_shape_with_tile_padding = shape.with_tile_padding();
        const auto new_height = new_shape_with_tile_padding[-2];
        const auto new_width = new_shape_with_tile_padding[-1];

        const auto is_tile_multiple = (new_height % ttnn::TILE_SIZE == 0 && new_width % ttnn::TILE_SIZE == 0);
        if (not is_tile_multiple) {
            return detail::row_major_reshape(tensor, shape);
        }

        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            if (tensor_shape.with_tile_padding()[-1] == new_width) {
                return tensor.reshape(shape.value);
            }
        } else {
            return tensor.reshape(shape.value);
        }
    }
    return detail::row_major_reshape(tensor, shape);
}



} // ttnn::operations::data_movement namespace
