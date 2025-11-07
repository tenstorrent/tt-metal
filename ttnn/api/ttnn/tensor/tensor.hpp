#pragma once
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/tensor/types.hpp>
#include "ttnn/tensor/shape/shape.hpp"
#include <ttnn/tensor/types.hpp>

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;
using TensorSpec = tt::tt_metal::TensorSpec;

std::string to_string(const tt::tt_metal::Tensor& tensor);

}  // namespace ttnn
