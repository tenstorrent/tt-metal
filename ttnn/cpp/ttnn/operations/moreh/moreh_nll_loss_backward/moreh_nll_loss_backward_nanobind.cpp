// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_nll_loss_backward.hpp"
#include "cpp/ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

void bind_moreh_nll_loss_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_nll_loss_backward,
        R"doc(moreh_nll_loss_backward(target_tensor: ttnn.Tensor, weight_tensor: Optional[ttnn.Tensor], output_grad_tensor: ttnn.Tensor, input_grad_tensor: Optional[ttnn.Tensor], ignore_index: int, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor
            Compute backward for nll_loss operation with reduction set to None
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("target_tensor"),
            nb::arg("output_grad_tensor"),
            nb::arg("reduction_mean"),
            nb::kw_only(),
            nb::arg("weight_tensor") = std::nullopt,
            nb::arg("input_grad_tensor") = std::nullopt,
            nb::arg("divisor_tensor") = std::nullopt,
            nb::arg("ignore_index") = -100,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
