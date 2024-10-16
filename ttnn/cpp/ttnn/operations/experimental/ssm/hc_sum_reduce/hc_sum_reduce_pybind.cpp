// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hc_sum_reduce.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace py = pybind11;

void bind_hc_sum_reduce(py::module& module) {
    using OperationType = decltype(ttnn::experimental::hc_sum_reduce);

    const auto doc = R"doc(
        Performs a custom reduction along dim 3 which is used in the SSM block of the Mamba architecture. Performs the following PyTorch equivalent (where latent_size = 32):
            x = torch.sum(x.reshape(1, 1, shape[2], shape[3] // latent_size, latent_size), dim=-1).reshape(1, 1, shape[2], shape[3] // latent_size)
    )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::hc_sum_reduce,
        doc,
        ttnn::pybind_overload_t{[](const OperationType& self,
                                   const ttnn::Tensor& input,
                                   const std::optional<MemoryConfig>& memory_config,
                                   const std::optional<DataType> dtype,
                                   const std::optional<MathFidelity> math_fidelity,
                                   uint8_t queue_id) {
                                    return self(queue_id, input, memory_config, dtype, math_fidelity);
                                },
                                py::arg("input"),
                                py::kw_only(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("dtype") = std::nullopt,
                                py::arg("math_fidelity") = std::nullopt,
                                py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::ssm::detail
