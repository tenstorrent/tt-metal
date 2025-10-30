#pragma once

namespace ttnn::experimental::jit {
class LazyTensor;

void evaluate(const LazyTensor& lazy_tensor);

}  // namespace ttnn::experimental::jit
