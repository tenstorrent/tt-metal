#pragma once

#include <memory>

namespace tt::tt_metal {
class Tensor;
}

namespace ttnn::experimental::lazy {

void evaluate(const tt::tt_metal::Tensor& tensor);

void print_graph(const tt::tt_metal::Tensor& tensor);
}  // namespace ttnn::experimental::lazy
