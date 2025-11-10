// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
void bind_moreh_bmm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_bmm_backward,
        "Moreh BMM Backward Operation",
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::moreh_bmm_backward)>& self,
               const ttnn::Tensor& output_grad,
               const ttnn::Tensor& input,
               const ttnn::Tensor& mat2,
               nb::object are_required_outputs_obj,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& mat2_grad,
               const std::optional<ttnn::MemoryConfig>& input_grad_memory_config,
               const std::optional<ttnn::MemoryConfig>& mat2_grad_memory_config,
               nb::object compute_kernel_config_obj) {
                // Accept tuple/list/sequence for are_required_outputs; default to [True, True]
                std::vector<bool> are_required_outputs{true, true};
                if (!are_required_outputs_obj.is_none()) {
                    try {
                        // Try direct cast first (handles lists)
                        are_required_outputs = nb::cast<std::vector<bool>>(are_required_outputs_obj);
                    } catch (const nb::cast_error&) {
                        // Fallback: handle tuple manually
                        if (nb::isinstance<nb::tuple>(are_required_outputs_obj)) {
                            nb::tuple t = nb::cast<nb::tuple>(are_required_outputs_obj);
                            are_required_outputs.clear();
                            are_required_outputs.reserve(t.size());
                            for (size_t i = 0; i < t.size(); ++i) {
                                are_required_outputs.push_back(nb::cast<bool>(t[i]));
                            }
                        } else {
                            throw nb::type_error("are_required_outputs must be a sequence of bools or None");
                        }
                    }
                }

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
                    output_grad,
                    input,
                    mat2,
                    are_required_outputs,
                    input_grad,
                    mat2_grad,
                    input_grad_memory_config,
                    mat2_grad_memory_config,
                    compute_kernel_config);
            },
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("mat2"),
            nb::arg("are_required_outputs") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("mat2_grad") = nb::none(),
            nb::arg("input_grad_memory_config") = nb::none(),
            nb::arg("mat2_grad_memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
