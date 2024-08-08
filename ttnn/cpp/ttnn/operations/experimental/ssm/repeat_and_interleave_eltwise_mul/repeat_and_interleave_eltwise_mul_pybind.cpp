// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "repeat_and_interleave_eltwise_mul.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace py = pybind11;

void bind_repeat_and_interleave_eltwise_mul(py::module& module) {
    using OperationType = decltype(ttnn::experimental::repeat_and_interleave_eltwise_mul);

    const auto doc = R"doc(
        Performs a special eltwise multiply for SSM models. Given tensor A with shape [1, 1, 32, 32] and tensor B with shape [1, 1, 32, W] where W is some multiple of 32, perform the following PyTorch equivalent:
            A.repeat(1, 1, 1, W) * B.repeat_interleave(32, dim=-1))doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::repeat_and_interleave_eltwise_mul,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               const std::optional<MathFidelity> math_fidelity,
               uint8_t queue_id) { return self(queue_id, a, b, memory_config, dtype, math_fidelity); },
            py::arg("a"),
            py::arg("b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::ssm::detail
