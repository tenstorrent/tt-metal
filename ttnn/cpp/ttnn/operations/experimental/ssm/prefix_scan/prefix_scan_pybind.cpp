// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefix_scan.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace py = pybind11;

void bind_prefix_scan(py::module& module) {
    using OperationType = decltype(ttnn::experimental::prefix_scan);

    const auto doc =
        R"doc(Performs a prefix scan to produce the SSM hidden states across an entire sequence. All input and output tensors are expected to be shape [1, 1, L, 2EN]. Values of 2EN and L can be any multiple of 32.)doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::prefix_scan,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& bx,
               const ttnn::Tensor& h_prev,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               const std::optional<MathFidelity> math_fidelity,
               uint8_t queue_id) { return self(queue_id, a, bx, h_prev, memory_config, dtype, math_fidelity); },
            py::arg("a"),
            py::arg("bx"),
            py::arg("h_prev"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::ssm::detail
