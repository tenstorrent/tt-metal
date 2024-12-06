// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/binary_ng.hpp"

namespace ttnn::operations::binary_ng {
namespace detail {
void bind_binary_ng_operation(py::module& module) {
    using OperationType = decltype(ttnn::experimental::add);

    bind_registered_operation(
        module,
        ttnn::experimental::add,
        "Binary Add Ng Operation",

        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, scalar, dtype, memory_config, output_tensor);
            },
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, input_tensor_b, dtype, memory_config, output_tensor);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}
}  // namespace detail

void py_module(py::module& module) { detail::bind_binary_ng_operation(module); }
}  // namespace ttnn::operations::eltwise::binary_ng
