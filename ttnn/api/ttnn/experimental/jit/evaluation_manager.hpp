#pragma once

#include <memory>

namespace ttnn::experimental::jit {
class LazyTensor;

void evaluate(const std::shared_ptr<LazyTensor>& lazy_tensor);

}  // namespace ttnn::experimental::jit
