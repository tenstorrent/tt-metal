// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "rmsnorm.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = nanobind;

void bind_normalization_rms_norm(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::rms_norm,
        R"doc(
            Compute rms_norm over :attr:`input_tensor`.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            epsilon (float): 1e-12.
            weight (ttnn.Tensor, optional): Defaults to `None`.
            bias (ttnn.Tensor, optional): Defaults to `None`.
            residual_input_tensor (ttnn.Tensor, optional): Defaults to `None`.
            program_config (ttnn.ProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = std::nullopt,
            nb::arg("bias") = std::nullopt,
            nb::arg("residual_input_tensor") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("program_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::normalization::detail
