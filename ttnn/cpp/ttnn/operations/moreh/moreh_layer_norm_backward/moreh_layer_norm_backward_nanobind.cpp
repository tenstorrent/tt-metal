// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
void bind_moreh_layer_norm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_layer_norm_backward,
        "Moreh Layer Norm Backward Operation",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::moreh_layer_norm_backward)& self,
               const Tensor& output_grad,
               const Tensor& input,
               const Tensor& mean,
               const Tensor& rstd,
               uint32_t normalized_dims,
               const std::optional<const Tensor>& gamma,
               const std::optional<const Tensor>& input_grad,
               const std::optional<const Tensor>& gamma_grad,
               const std::optional<const Tensor>& beta_grad,
               const std::optional<MemoryConfig>& memory_config,
               nb::object compute_kernel_config_obj) {
                std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
                if (!compute_kernel_config_obj.is_none()) {
                    if (nb::isinstance<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj);
                    } else if (nb::isinstance<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj);
                    } else {
                        throw nb::type_error(
                            "compute_kernel_config must be WormholeComputeKernelConfig | "
                            "GrayskullComputeKernelConfig or None");
                    }
                }
                return self(
                    output_grad,
                    input,
                    mean,
                    rstd,
                    normalized_dims,
                    gamma,
                    input_grad,
                    gamma_grad,
                    beta_grad,
                    memory_config,
                    compute_kernel_config);
            },
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("mean"),
            nb::arg("rstd"),
            nb::arg("normalized_dims"),
            nb::kw_only(),
            nb::arg("gamma") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("gamma_grad") = nb::none(),
            nb::arg("beta_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
