// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
void bind_moreh_layer_norm_backward_operation(nb::module_& mod) {
    const auto* doc = "Moreh Layer Norm Backward Operation";

    ttnn::bind_function<"moreh_layer_norm_backward">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                uint32_t,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<ttnn::MemoryConfig>&,
                const std::optional<ttnn::DeviceComputeKernelConfig>&>(&ttnn::moreh_layer_norm_backward),
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
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
