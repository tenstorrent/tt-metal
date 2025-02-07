// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/binary_ng.hpp"
#include "ttnn/operations/eltwise/binary_ng/hypot/hypot.hpp"

namespace ttnn::operations::binary_ng {
namespace detail {
template <typename T>
void bind_binary_ng_operation(py::module& module, T op, const std::string& docstring) {
    bind_registered_operation(
        module,
        op,
        docstring,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& post_activations,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryWithParam>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& post_activations,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_binary_ng_bitwise_ops(py::module& module, T op, const std::string& docstring) {
    bind_registered_operation(
        module,
        op,
        docstring,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, scalar, memory_config, output_tensor);
            },
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, input_tensor_b, memory_config, output_tensor);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_hypot(
    py::module& module,
    const T& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - {5}

            {8}

        Example:
            >>> tensor1 = {6}
            >>> tensor2 = {7}
            >>> output = {1}(tensor1, tensor2)
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const T& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<Tensor>& output_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor_a, input_tensor_b, memory_config, output_tensor);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename T>
void bind_inplace_binary_ng_operation(py::module& module, T op, const std::string& docstring) {
    bind_registered_operation(
        module,
        op,
        docstring,

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const ttnn::SmallVector<unary::UnaryWithParam>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& post_activations,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, scalar, lhs_activations, rhs_activations, post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("queue_id") = DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::UnaryWithParam>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryWithParam>& post_activations,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id, input_tensor_a, input_tensor_b, lhs_activations, rhs_activations, post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryWithParam>(),
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary_ng_operation(module, ttnn::experimental::add, "Binary Add Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::sub, "Binary Sub Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::mul, "Binary Mul Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::div, "Binary Div Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::rsub, "Binary Rsub Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::pow, "Binary Power Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::gt, "Binary Greater Than Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::lt, "Binary Less Than Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::lte, "Binary Less Than or Equal To Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::gte, "Binary Greater Than or Equal To Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::eq, "Binary Equal Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::ne, "Binary Not Equal Operation");
    detail::bind_binary_ng_operation(
        module, ttnn::experimental::squared_difference, "Binary Squared Difference Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::bias_gelu, "Binary Bias GELU Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::logical_and, "Binary Logical And Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::logical_or, "Binary Logical Or Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::logical_xor, "Binary Logical Xor Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::ldexp, "Binary Ldexp Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::logaddexp, "Binary Logaddexp Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::logaddexp2, "Binary Logaddexp2 Operation");

    detail::bind_binary_ng_bitwise_ops(module, ttnn::experimental::bitwise_and, "Binary bitwise_and Operation");
    detail::bind_binary_ng_bitwise_ops(module, ttnn::experimental::bitwise_xor, "Binary bitwise_xor Operation");
    detail::bind_binary_ng_bitwise_ops(module, ttnn::experimental::bitwise_or, "Binary bitwise_or Operation");
    detail::bind_binary_ng_bitwise_ops(
        module, ttnn::experimental::bitwise_left_shift, "Binary bitwise_left_shift Operation");
    detail::bind_binary_ng_bitwise_ops(
        module, ttnn::experimental::bitwise_right_shift, "Binary bitwise_right_shift Operation");

    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::add_, "Binary Add In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::sub_, "Binary Subtract In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::mul_, "Binary Multiply In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::div_, "Binary Divide In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::rsub_, "Binary Rsub In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::pow_, "Binary Power In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::gt_, "Binary Greater Than In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::lt_, "Binary Less Than In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::lte_, "Binary Less Than or Equal To In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::gte_, "Binary Greater Than or Equal To In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::eq_, "Binary Equal In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::ne_, "Binary Not Equal In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::squared_difference_, "Binary Squared Difference In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::bias_gelu_, "Binary Bias GELU In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::logical_and_, "Binary Logical And In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::logical_or_, "Binary Logical Or In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::logical_xor_, "Binary Logical Xor In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::ldexp_, "Binary Ldexp In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::logaddexp_, "Binary Logaddexp In-place Operation");
    detail::bind_inplace_binary_ng_operation(
        module, ttnn::experimental::logaddexp2_, "Binary Logaddexp2 In-place Operation");
    detail::bind_hypot(
        module,
        ttnn::experimental::hypot,
        R"doc(Computes hypot :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \sqrt{(\mathrm{input\_tensor\_a}_i^2 + \mathrm{input\_tensor\_b}_i^2)})doc",
        R"doc(BFLOAT16)doc");
}
}  // namespace ttnn::operations::binary_ng
