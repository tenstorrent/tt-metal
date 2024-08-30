// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "pybind11/cast.h"
#include "pybind11/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_adam {
void bind_moreh_adam_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_adam,
        "Moreh Adam Operation",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::moreh_adam)& self,
               const Tensor& param_in,
               const Tensor& grad,
               const Tensor& exp_avg_in,
               const Tensor& exp_avg_sq_in,

               float lr,
               float beta1,
               float beta2,
               float eps,
               float weight_decay,
               uint32_t step,
               bool amsgrad,

               const std::optional<const Tensor> max_exp_avg_sq_in,
               const std::optional<const Tensor> param_out,
               const std::optional<const Tensor> exp_avg_out,
               const std::optional<const Tensor> exp_avg_sq_out,
               const std::optional<const Tensor> max_exp_avg_sq_out,

               const MemoryConfig& memory_config,
               const DeviceComputeKernelConfig compute_kernel_config) {
                return self(
                    param_in,
                    grad,
                    exp_avg_in,
                    exp_avg_sq_in,

                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                    amsgrad,

                    max_exp_avg_sq_in,
                    param_out,
                    exp_avg_out,
                    exp_avg_sq_out,
                    max_exp_avg_sq_out,

                    memory_config,
                    compute_kernel_config);
            },
            py::arg("param_in"),
            py::arg("grad"),
            py::arg("exp_avg_in"),
            py::arg("exp_avg_sq_in"),

            py::arg("lr"),
            py::arg("beta1"),
            py::arg("beta2"),
            py::arg("eps"),
            py::arg("weight_decay"),
            py::arg("step"),
            py::arg("amsgrad"),

            py::arg("max_exp_avg_sq_in") = std::nullopt,
            py::arg("param_out") = std::nullopt,
            py::arg("exp_avg_out") = std::nullopt,
            py::arg("exp_avg_sq_out") = std::nullopt,
            py::arg("max_exp_avg_sq_out") = std::nullopt,

            py::arg("memory_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_adam
