#pragma once

#include <memory>

namespace tt::tt_metal {
class Tensor;
}

namespace ttnn::experimental::lazy {
class LazyTensor;

void evaluate(const std::shared_ptr<LazyTensor>& lazy_tensor);

}  // namespace ttnn::experimental::lazy
