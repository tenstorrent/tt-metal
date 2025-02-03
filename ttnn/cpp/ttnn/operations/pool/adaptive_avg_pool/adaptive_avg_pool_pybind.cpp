// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_avg_pool_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/decorators.hpp"
#include "ttnn/operations/pool/adaptive_avg_pool/adaptive_avg_pool.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pool {
namespace detail {
template <typename pool_operation_t>
void bind_adaptive_avg_pool(pybind11::module& module, const pool_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(operation)& self,
               const ttnn::Tensor& input,
               const ttnn::Shape& output_size,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input, output_size, memory_config);
            },
            pybind11::arg("input").noconvert(),
            pybind11::arg("output_size"),
            pybind11::kw_only(),
            pybind11::arg("memory_config") = std::nullopt});
}
}  // namespace detail

void py_bind_adaptive_avg_pool(pybind11::module& module) {
    detail::bind_adaptive_avg_pool(
        module,
        ttnn::adaptive_avg_pool2d,
        R"doc(adaptive_avg_pool2d(input: ttnn.Tensor, output_size: ttnn.Shape) -> ttnn.Tensor
        Applies a 2D adaptive average pooling operation on the input tensor.
        Args:
            * :attr:`input`: Input tensor.
            * :attr:`output_size`: Target output size.
            * :attr:`<optional> mem_config`.
        )doc");

    detail::bind_adaptive_avg_pool(
        module,
        ttnn::adaptive_avg_pool1d,
        R"doc(adaptive_avg_pool1d(input: ttnn.Tensor, output_size: ttnn.Shape) -> ttnn.Tensor
        Applies a 1D adaptive average pooling operation on the input tensor.
        Args:
            * :attr:`input`: Input tensor.
            * :attr:`output_size`: Target output size.
            * :attr:`<optional> mem_config`.
        )doc");
}
}  // namespace ttnn::operations::pool
