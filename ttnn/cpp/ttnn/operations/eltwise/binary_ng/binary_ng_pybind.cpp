// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/binary_ng.hpp"

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
               const ttnn::SmallVector<unary::UnaryOpType>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& post_activations,
               const uint8_t& queue_id) -> ttnn::Tensor {
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
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("queue_id") = 0},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::UnaryOpType>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& post_activations,
               uint8_t queue_id) -> ttnn::Tensor {
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
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
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
               const std::optional<const DataType>& dtype,
               const ttnn::SmallVector<unary::UnaryOpType>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& post_activations,
               const uint8_t& queue_id) -> ttnn::Tensor {
                return self(
                    queue_id, input_tensor_a, scalar, dtype, lhs_activations, rhs_activations, post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("queue_id") = 0},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const ttnn::SmallVector<unary::UnaryOpType>& lhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& rhs_activations,
               const ttnn::SmallVector<unary::UnaryOpType>& post_activations,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("lhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("rhs_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("post_activations") = ttnn::SmallVector<unary::UnaryOpType>(),
            py::arg("queue_id") = 0});
}
}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary_ng_operation(module, ttnn::experimental::add, "Binary Add Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::sub, "Binary Sub Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::mul, "Binary Mul Operation");
    detail::bind_binary_ng_operation(module, ttnn::experimental::div, "Binary Div Operation");
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
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::add_, "Binary Add In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::sub_, "Binary Subtract In-place Operation");
    detail::bind_inplace_binary_ng_operation(module, ttnn::experimental::mul_, "Binary Multiply In-place Operation");
}
}  // namespace ttnn::operations::binary_ng
