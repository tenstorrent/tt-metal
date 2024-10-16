// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_pybind.hpp"

#include <pybind11/pybind11.h>

#include "moreh_nll_loss.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_nll_loss {

void bind_moreh_nll_loss_operation(py::module &module) {
    bind_registered_operation(
        module,
        ttnn::moreh_nll_loss,
        R"doc(moreh_nll_loss(input_tensor: ttnn.Tensor, target_tensor: ttnn.Tensor, reduction: string, weight_tensor: Optional[ttnn.Tensor], divisor_tensor: Optional[ttnn.Tensor], output_tensor: Optional[ttnn.Tensor], ignore_index: int, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor
            Compute backward for nll_loss operation with reduction set to None
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                 py::arg("target_tensor"),
                                 py::arg("reduction"),
                                 py::kw_only(),
                                 py::arg("weight_tensor") = std::nullopt,
                                 py::arg("divisor_tensor") = std::nullopt,
                                 py::arg("output_tensor") = std::nullopt,
                                 py::arg("ignore_index") = -100,
                                 py::arg("memory_config") = std::nullopt,
                                 py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
