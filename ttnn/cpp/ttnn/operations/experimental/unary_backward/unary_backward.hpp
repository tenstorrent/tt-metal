// gelu_backward.hpp
#pragma once

namespace ttnn::operations::experimental {

struct GeluBackwardOperation {
    static Tensor invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const string& parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto gelu_bw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gelu_bw",
    ttnn::operations::experimental::GeluBackwardOperation>();
}
