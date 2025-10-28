#pragma once

#include <tt-metalium/tensor/tensor_impl.hpp>

namespace ttnn {

template <typename T>
std::string to_string(const tt::tt_metal::Tensor& tensor);

}  // namespace ttnn
