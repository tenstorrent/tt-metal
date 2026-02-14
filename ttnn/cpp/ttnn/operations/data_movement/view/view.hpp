// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

/**
 * @brief Performs a zero-cost view operation that returns the same tensor with a new shape.
 *
 * This operation provides a view of the tensor with a different shape without copying data.
 * The following conditions must be met:
 * - The memory must be stored on the device
 * - The last dimension must not change
 * - In TILE layout, the second last two dimensions must not change OR there is no padding
 *   on the second last dimension
 *
 * @param input_tensor The input tensor to view
 * @param logical_shape The new logical shape for the tensor
 * @return A tensor with the new shape that shares storage with the input
 */
ttnn::Tensor view(const ttnn::Tensor& input_tensor, const ttnn::Shape& logical_shape);

/**
 * @brief Performs a zero-cost view operation with shape inference.
 *
 * This overload accepts a shape vector and infers dimensions (supporting -1 for automatic sizing).
 *
 * @param input_tensor The input tensor to view
 * @param shape_vector Vector of dimensions, where -1 can be used for automatic dimension inference
 * @return A tensor with the new shape that shares storage with the input
 */
ttnn::Tensor view(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector);

}  // namespace ttnn
