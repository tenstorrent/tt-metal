// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "moreh_matmul.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_matmul {
void bind_moreh_matmul_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_matmul,
        "Moreh Matmul Operation",
        ttnn::nanobind_overload_t{
            [](decltype(ttnn::moreh_matmul)& self,
               const ttnn::Tensor& input,
               const ttnn::Tensor& other,
               const bool transpose_input,
               const bool transpose_other,
               const std::optional<ttnn::Tensor>& output,
               const std::optional<const ttnn::Tensor>& bias,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               nb::object compute_kernel_config_obj) -> ttnn::Tensor {
                std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
                if (!compute_kernel_config_obj.is_none()) {
                    if (nb::isinstance<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj);
                    } else if (nb::isinstance<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj);
                    } else {
                        nb::type_error(
                            "compute_kernel_config must be WormholeComputeKernelConfig, GrayskullComputeKernelConfig, "
                            "or None");
                    }
                }
                return self(
                    input,
                    other,
                    transpose_input,
                    transpose_other,
                    output,
                    bias,
                    memory_config,
                    compute_kernel_config);  // forwards optional<std::variant<...>>
            },
            nb::arg("input"),
            nb::arg("other"),
            nb::kw_only(),
            nb::arg("transpose_input") = false,
            nb::arg("transpose_other") = false,
            nb::arg("output") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_matmul
