// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {
void bind_moreh_layer_norm_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_layer_norm,
        "Moreh Layer Norm Operation",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::moreh_layer_norm)& self,
               const Tensor& input,
               uint32_t normalized_dims,
               float eps,
               const std::optional<const Tensor>& gamma,
               const std::optional<const Tensor>& beta,
               const std::optional<const Tensor>& output,
               const std::optional<const Tensor>& mean,
               const std::optional<const Tensor>& rstd,
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
                    input, normalized_dims, eps, gamma, beta, output, mean, rstd, memory_config, compute_kernel_config);
            },
            nb::arg("input"),
            nb::arg("normalized_dims"),
            nb::arg("eps") = 1e-5f,
            nb::arg("gamma") = nb::none(),
            nb::arg("beta") = nb::none(),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("mean") = nb::none(),
            nb::arg("rstd") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
