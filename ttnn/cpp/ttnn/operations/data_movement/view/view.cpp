// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    auto layout = tensor.layout();
    auto tensor_shape = tensor.logical_shape();
    // First Case, No reshape Required
    if (tensor_shape == shape) {
        return tensor;
    }

    const uint32_t tile_first_dim = tensor.tensor_spec().tile().get_width();
    const uint32_t tile_second_dim = tensor.tensor_spec().tile().get_height();
    const uint32_t shape_second_last_dim = shape.rank() >= 2 ? shape[-2] : 1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;
    // Validate the operation
    TT_FATAL(
        shape.volume() == tensor.logical_volume(),
        "Invalid view, logical volumes are changing from {} to {}",
        tensor.logical_volume(),
        shape.volume());
    TT_FATAL(
        ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE),
        "View requires the tensor be stored on device, use reshape instead");
    TT_FATAL(
        (tensor_shape[-1] == shape[-1]),
        "The last dimension can not change in view, attempting to change last dimension from {} to {}, use reshape "
        "instead",
        tensor_shape[-1],
        shape[-1]);
    TT_FATAL(
        (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) ||                  // Its row major
            (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
            ((shape_second_last_dim % tile_second_dim == 0) && (tensor_shape_second_last_dim % tile_second_dim == 0)),
        "Invalid second last dims for TILED reshape, from {} to {}, use reshape instead\n",
        tensor_shape_second_last_dim,
        shape_second_last_dim);
    // Perform the View
    return PerformView(tensor, shape, shape, tile_first_dim, tile_second_dim);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, tt::stl::Span<const int32_t> shape_vector) {
    return invoke(tensor, tt::tt_metal::infer_dims_for_reshape(tensor, shape_vector));
}

}  // namespace ttnn::operations::data_movement
