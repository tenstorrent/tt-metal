#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn {

Tensor tensor_reshape(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape);
Tensor tensor_reshape(const Tensor& input_tensor, const Shape& new_shape);

}  // namespace ttnn
