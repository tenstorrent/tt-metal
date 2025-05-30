// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {

void bind_moreh_cumsum_operation(py::module& module) {
    using OperationType = decltype(ttnn::experimental::cumsum);

    bind_registered_operation(
        module,
        ttnn::moreh_cumsum,
        "Moreh Cumsum Operation",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const int64_t dim,
               std::optional<Tensor> output,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(DefaultQueueId, input, dim, input.dtype(), output, false, memory_config);
            },
            py::arg("input"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

void bind_moreh_cumsum_backward_operation(py::module& module) {
    using OperationType = decltype(ttnn::experimental::cumsum_backward);

    bind_registered_operation(
        module,
        ttnn::moreh_cumsum_backward,
        "Moreh Cumsum Backward Operation",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& output_grad,
               const int64_t dim,
               std::optional<Tensor> input_grad,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(DefaultQueueId, output_grad, dim, output_grad.dtype(), input_grad, memory_config);
            },
            py::arg("output_grad"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("input_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::moreh::moreh_cumsum
