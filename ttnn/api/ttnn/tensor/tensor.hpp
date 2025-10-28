#pragma once
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/tensor/types.hpp>
#include "ttnn/tensor/shape/shape.hpp"
#include <ttnn/tensor/types.hpp>

namespace ttnn {

std::string write_to_string(const Tensor& tensor);
void tensor_print(const Tensor& input_tensor);
}  // namespace ttnn
