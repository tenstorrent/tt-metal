#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"

namespace ttnn {
std::string write_to_string(const Tensor& tensor) { return tt::tt_metal::tensor_impl::to_string(tensor); }

void tensor_print(const Tensor& input_tensor) { input_tensor.print(); }
}  // namespace ttnn
