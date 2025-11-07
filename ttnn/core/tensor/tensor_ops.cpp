#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/core/core.hpp"

using namespace tt::tt_metal;

Tensor tensor_reshape(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    return ttnn::reshape(input_tensor, new_logical_shape, new_padded_shape);
}

Tensor tensor_reshape(const Tensor& input_tensor, const Shape& new_shape) {
    return ttnn::reshape(input_tensor, new_shape);
}
