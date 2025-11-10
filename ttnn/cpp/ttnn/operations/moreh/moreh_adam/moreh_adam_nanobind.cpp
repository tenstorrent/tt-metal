// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam.hpp"

namespace ttnn::operations::moreh::moreh_adam {
void bind_moreh_adam_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_adam,
        "Moreh Adam Operation",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::moreh_adam)& self,
               const Tensor& param_in,
               const Tensor& grad,
               const Tensor& exp_avg_in,
               const Tensor& exp_avg_sq_in,
               std::optional<float> lr,
               std::optional<float> beta1,
               std::optional<float> beta2,
               std::optional<float> eps,
               std::optional<float> weight_decay,
               std::optional<uint32_t> step,
               std::optional<bool> amsgrad,
               const std::optional<const Tensor>& max_exp_avg_sq_in,
               const std::optional<const Tensor>& param_out,
               const std::optional<const Tensor>& exp_avg_out,
               const std::optional<const Tensor>& exp_avg_sq_out,
               const std::optional<const Tensor>& max_exp_avg_sq_out,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               nb::object compute_kernel_config_obj) {
                std::optional<DeviceComputeKernelConfig> compute_kernel_config;
                if (!compute_kernel_config_obj.is_none()) {
                    if (nb::isinstance<WormholeComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<WormholeComputeKernelConfig>(compute_kernel_config_obj);
                    } else if (nb::isinstance<GrayskullComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<GrayskullComputeKernelConfig>(compute_kernel_config_obj);
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
            nb::kw_only(),
            nb::arg("lr") = 0.001f,
            nb::arg("beta1") = 0.9f,
            nb::arg("beta2") = 0.999f,
            nb::arg("eps") = 1e-8f,
            nb::arg("weight_decay") = 0.0f,
            nb::arg("step") = 0,
            nb::arg("amsgrad") = false,
            nb::arg("max_exp_avg_sq_in") = nb::none(),
            nb::arg("param_out") = nb::none(),
            nb::arg("exp_avg_out") = nb::none(),
            nb::arg("exp_avg_sq_out") = nb::none(),
            nb::arg("max_exp_avg_sq_out") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_adam
