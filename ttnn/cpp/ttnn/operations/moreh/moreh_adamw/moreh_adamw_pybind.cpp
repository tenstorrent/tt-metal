// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adamw_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adamw/moreh_adamw.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_adamw {

void bind_moreh_adamw_operation(py::module& module) {
    bind_registered_operation(module,
                              ttnn::moreh_adamw,
                              R"doc(
            moreh_adamw(
                param_in: ttnn.Tensor,
                grad: ttnn.Tensor,
                exp_avg_in: ttnn.Tensor,
                exp_avg_sq_in: ttnn.Tensor,
                lr: Optional[float] = 0.001,
                beta1: Optional[float] = 0.9,
                beta2: Optional[float] = 0.999,
                eps: Optional[float] = 1e-8,
                weight_decay: Optional[float] = 1e-2,
                step: Optional[int] = 0,
                amsgrad: Optional[bool] = false,
                max_exp_avg_sq_in: Optional[ttnn.Tensor],
                param_out: Optional[ttnn.Tensor],
                exp_avg_out: Optional[ttnn.Tensor],
                exp_avg_sq_out: Optional[ttnn.Tensor],
                max_exp_avg_sq_out: Optional[ttnn.Tensor],
                memory_config: Optional[ttnn.MemoryConfig] = None,
                compute_kernel_config: Optional[DeviceComputeKernelConfig]
            ) -> ttnn.Tensor
            Compute backward for nll_loss operation with reduction set to None
        )doc",
                              ttnn::pybind_arguments_t{py::arg("param_in"),
                                                       py::arg("grad"),
                                                       py::arg("exp_avg_in"),
                                                       py::arg("exp_avg_sq_in"),
                                                       py::arg("lr") = 0.001f,
                                                       py::arg("beta1") = 0.9f,
                                                       py::arg("beta2") = 0.999f,
                                                       py::arg("eps") = 1e-8f,
                                                       py::arg("weight_decay") = 1e-2f,
                                                       py::arg("step") = 0,
                                                       py::arg("amsgrad") = false,
                                                       py::kw_only(),

                                                       py::arg("max_exp_avg_sq_in") = std::nullopt,
                                                       py::arg("param_out") = std::nullopt,
                                                       py::arg("exp_avg_out") = std::nullopt,
                                                       py::arg("exp_avg_sq_out") = std::nullopt,
                                                       py::arg("max_exp_avg_sq_out") = std::nullopt,

                                                       py::arg("memory_config") = std::nullopt,
                                                       py::arg("compute_kernel_config") = std::nullopt});
}

void py_module(py::module& module) {
    bind_moreh_adamw_operation(module);
}

}  // namespace ttnn::operations::moreh::moreh_adamw
