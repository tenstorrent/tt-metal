// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_nll_loss.hpp"
#include "cpp/ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_nll_loss {

void bind_moreh_nll_loss_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_nll_loss,
        R"doc(moreh_nll_loss(input_tensor: ttnn.Tensor, target_tensor: ttnn.Tensor, reduction: string, weight_tensor: Optional[ttnn.Tensor], divisor_tensor: Optional[ttnn.Tensor], output_tensor: Optional[ttnn.Tensor], ignore_index: int, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor
            Compute backward for nll_loss operation with reduction set to None
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("target_tensor"),
            nb::arg("reduction"),
            nb::kw_only(),
            nb::arg("weight_tensor") = std::nullopt,
            nb::arg("divisor_tensor") = std::nullopt,
            nb::arg("output_tensor") = std::nullopt,
            nb::arg("ignore_index") = -100,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
