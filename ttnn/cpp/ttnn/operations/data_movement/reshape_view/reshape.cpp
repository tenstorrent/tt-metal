// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <functional>
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
    if (!ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
        return tensor.reshape(shape);
    }
    auto tensor_shape = tensor.shape();
    auto layout = tensor.layout();
    auto device = tensor.device();
    auto memory_config = tensor.memory_config();
    auto host_tensor = tensor.cpu();
    auto rm_tensor = ttnn::to_layout(host_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
    if(tensor_shape.has_tile_padding()) {
        ttnn::Tensor slice_input;
        std::vector<uint32_t> begins;
        std::vector<uint32_t> ends;
        TT_FATAL(tensor_shape.rank() <= 4, "Only up to 4D tensors");
        auto host_tensor_4d = unsqueeze_to_4D(rm_tensor);
        auto tensor_shape_4d = host_tensor_4d.shape();
        begins = std::vector<uint32_t>({0, 0, 0, 0});
        ends = std::vector<uint32_t>({tensor_shape_4d[0], tensor_shape_4d[1], tensor_shape_4d[2], tensor_shape_4d[3]});
        auto step = std::vector<uint32_t>({1, 1, 1, 1});
        host_tensor_4d = ttnn::slice(host_tensor_4d, begins, ends, step, std::nullopt);
        host_tensor = squeeze_from_4D(host_tensor_4d, tensor_shape.rank());
    }
    auto host_reshape_tensor = rm_tensor.reshape(shape);
    auto final_layout_tensor = ttnn::to_layout(host_reshape_tensor, layout, std::nullopt, std::nullopt, (Device *)nullptr);
    auto device_tensor = ttnn::data_transfer_to_device(final_layout_tensor, device, memory_config);
    return device_tensor;
}

ttnn::Tensor row_major_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    const auto layout = tensor.get_layout();
    auto shape_with_padding = shape.with_tile_padding();
    auto tensor_shape = tensor.get_shape();
    auto tensor_shape_with_padding = tensor_shape.with_tile_padding();

    //Constraint in device kernel
    uint32_t ROW_MAJOR_WIDTH = 8;
    ttnn::Tensor reshaped_rm_tensor;
    if((tensor_shape[-1] % ROW_MAJOR_WIDTH == 0 && shape[-1] % ROW_MAJOR_WIDTH == 0)) {
        auto rm_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
        if (rm_tensor.is_contiguous()) {
            // Page size depends on the width, so only modify the shape if the width is the same
            if (tensor_shape_with_padding[-1] == shape_with_padding[-1]) {
                return rm_tensor.reshape(shape);
            }
            //Different page width, going to use device kernel that does transpose
            else {
                auto original_rank = shape.rank();
                auto tensor_4d = unsqueeze_to_4D(rm_tensor);
                const auto shape_4d = shape.to_rank(4);
                auto reshaped_tensor = ttnn::reshape_on_device(tensor_4d, shape_4d[0], shape_4d[1], shape_4d[2], shape_4d[3], tensor.memory_config());
                reshaped_rm_tensor = squeeze_from_4D(reshaped_tensor, original_rank);
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape_with_padding[-1] == shape_with_padding[-1] and
                tensor_shape_with_padding[-2] == shape_with_padding[-2]) {
                reshaped_rm_tensor = rm_tensor.reshape(shape);
            }
        } else {
            reshaped_rm_tensor = host_reshape(tensor, shape);
        }

    }
    // Can'd do untilize on device due to inner dim size
    else {
        reshaped_rm_tensor = host_reshape(tensor, shape);
    }

    if (((shape[-1] * tensor.element_size()) % sizeof(uint32_t) == 0) and reshaped_rm_tensor.layout() != layout) {
        return ttnn::to_layout(reshaped_rm_tensor, layout, std::nullopt, std::nullopt, (Device *)nullptr);
    }
    else {
        return reshaped_rm_tensor;
    }
}

}


ttnn::Tensor ReshapeViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_shape();

    // First Case, No reshape Required
    if (tensor_shape == shape) {
        return tensor;
    }

    bool tile_tensor_view_reshape_possible = (layout == ttnn::Layout::TILE and
        ((shape.with_tile_padding()[-2] % ttnn::TILE_SIZE == 0) and (shape.with_tile_padding()[-1] % ttnn::TILE_SIZE == 0)) and
        (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1])
        );

    // For Tensors already on host we can do the tensor.reshape (changing of view)
    if (!(ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) or tile_tensor_view_reshape_possible) {
        return tensor.reshape(shape);
    }

    // Catch-all
    // Do the reshape in row-major
    return detail::row_major_reshape(tensor, shape);
}

ttnn::Tensor ReshapeViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::SimpleShape& shape) {
    return invoke(tensor, ttnn::Shape(shape.as_vector()));
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const std::vector<int32_t> & shape_vector
    ) {
    return invoke(tensor, tt::tt_metal::infer_dims_for_reshape(shape_vector, tensor.get_logical_volume()));
}

} // ttnn::operations::data_movement namespace
