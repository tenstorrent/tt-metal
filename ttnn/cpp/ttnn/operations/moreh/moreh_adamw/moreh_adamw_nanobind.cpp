// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adamw_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adamw/moreh_adamw.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

void bind_moreh_adamw_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
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
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::moreh_adamw)>& self,
               const ttnn::Tensor& param_in,
               const ttnn::Tensor& grad,
               const ttnn::Tensor& exp_avg_in,
               const ttnn::Tensor& exp_avg_sq_in,
               std::optional<float> lr,
               std::optional<float> beta1,
               std::optional<float> beta2,
               std::optional<float> eps,
               std::optional<float> weight_decay,
               std::optional<uint32_t> step,
               std::optional<bool> amsgrad,
               const std::optional<ttnn::Tensor>& max_exp_avg_sq_in,
               const std::optional<ttnn::Tensor>& param_out,
               const std::optional<ttnn::Tensor>& exp_avg_out,
               const std::optional<ttnn::Tensor>& exp_avg_sq_out,
               const std::optional<ttnn::Tensor>& max_exp_avg_sq_out,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               nb::object compute_kernel_config_obj) {
                std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
                if (!compute_kernel_config_obj.is_none()) {
                    if (nb::isinstance<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj);
                    } else if (nb::isinstance<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj);
                    } else {
                        throw nb::type_error(
                            "compute_kernel_config must be WormholeComputeKernelConfig | GrayskullComputeKernelConfig "
                            "| None");
                    }
                }
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
            nb::arg("param_in"),
            nb::arg("grad"),
            nb::arg("exp_avg_in"),
            nb::arg("exp_avg_sq_in"),
            nb::arg("lr") = 0.001f,
            nb::arg("beta1") = 0.9f,
            nb::arg("beta2") = 0.999f,
            nb::arg("eps") = 1e-8f,
            nb::arg("weight_decay") = 1e-2f,
            nb::arg("step") = 0,
            nb::arg("amsgrad") = false,
            nb::kw_only(),
            nb::arg("max_exp_avg_sq_in") = nb::none(),
            nb::arg("param_out") = nb::none(),
            nb::arg("exp_avg_out") = nb::none(),
            nb::arg("exp_avg_sq_out") = nb::none(),
            nb::arg("max_exp_avg_sq_out") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

void py_module(nb::module_& mod) { bind_moreh_adamw_operation(mod); }

}  // namespace ttnn::operations::moreh::moreh_adamw
