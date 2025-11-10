// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_nanobind.hpp"

#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "uniform.hpp"

namespace ttnn::operations::uniform {
void bind_uniform_operation(nb::module_& mod) {
    std::string doc =
        R"doc(
        Update in-place the input tensor with values drawn from the continuous uniform distribution 1 / (`to` - `from`).

        Args:
            input (ttnn.Tensor): The tensor that provides the shape for the generated uniform tensor.
            from (float32): The lower bound of the uniform distribution. Defaults to 0.
            to (float32): The upper bound of the uniform distribution. Defaults to 1.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for the compute kernel. Defaults to `None`.

        Returns:
            ttnn.Tensor: The `input` tensor with updated values drawn from the specified uniform distribution.

        Example:
            >>> input = ttnn.to_device(ttnn.from_torch(torch.ones(3, 3), dtype=torch.bfloat16)), device=device)
            >>> ttnn.uniform(input)

        )doc";

    bind_registered_operation(
        mod,
        ttnn::uniform,
        doc,
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::uniform)> self,
               const ttnn::Tensor& input,
               float from,
               float to,
               uint32_t seed,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               nb::object compute_kernel_config_obj) -> ttnn::Tensor {
                std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
                if (!compute_kernel_config_obj.is_none()) {
                    if (nb::isinstance<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::WormholeComputeKernelConfig>(compute_kernel_config_obj);
                    } else if (nb::isinstance<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj)) {
                        compute_kernel_config = nb::cast<ttnn::GrayskullComputeKernelConfig>(compute_kernel_config_obj);
                    } else {
                        throw nb::type_error(
                            "compute_kernel_config must be WormholeComputeKernelConfig or "
                            "GrayskullComputeKernelConfig");
                    }
                }
                return self(input, from, to, seed, memory_config, compute_kernel_config);
            },
            nb::arg("input"),
            nb::arg("from") = 0,
            nb::arg("to") = 1,
            nb::arg("seed") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
        });
}
}  // namespace ttnn::operations::uniform
