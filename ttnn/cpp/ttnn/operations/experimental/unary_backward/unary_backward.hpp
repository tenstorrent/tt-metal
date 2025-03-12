#pragma once

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental {

struct GeluBackwardOperation {
    static OptionalTensors invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const string& parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static OptionalTensors create_async_optional_output_tensors(
        QueueId queue_id,
        const Tensor& grad_output_tensor,
        const Tensor& input_tensor,
        const string& approximate,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad_tensor = std::nullopt);
};

}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto gelu_bw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gelu_bw",
    ttnn::operations::experimental::GeluBackwardOperation>();
}
