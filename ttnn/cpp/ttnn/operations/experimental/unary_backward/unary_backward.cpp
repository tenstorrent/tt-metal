// gelu_backward.cpp
#include "device/gelu_backward/gelu_backward_device_operation.hpp"
#include "unary_backward.hpp"

namespace ttnn::operations::experimental {

Tensor GeluBackwardOperation::invoke(
    const Tensor& grad_output_tensor,
    const Tensor& input_tensor,
    const string& approximate,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> input_grad_tensor) {
    // TODO: verify those attributes are calculated properly
    DataType output_dtype = input_tensor.get_dtype();
    auto arch = input_tensor.device()->arch();
    auto output_memory_config = input_grad_tensor.has_value() ? input_grad_tensor.value().memory_config()
                                                              : memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::gelu_bw(
        grad_output_tensor, input_tensor, approximate, output_dtype, output_memory_config, input_grad_tensor);
}

}  // namespace ttnn::operations::experimental
