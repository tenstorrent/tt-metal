// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "clone.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement::clone {
void bind_clone_operation(nb::module_& mod) {
    auto doc = R"doc(
        Clones the input tensor, creating a copy with the specified memory configuration and converting its data type to dtype. This operation does not alter the tensor's layout.

        Args:
            input (ttnn.Tensor): the input tensor to be cloned.

        Keyword Args:
            dtype (ttnn.DataType, optional): the target data type of the cloned tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the clone, options include DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): the configuration for the compute kernel. Defaults to `None`.

        Returns:
            ttnn.Tensor: the cloned output tensor.

        Note:
            * ROW_MAJOR_LAYOUT: Returns the tensor unpadded in the last two dimensions.
            * TILE_LAYOUT: Pads the tensor to ensure its width and height are multiples of 32.
            * If the input's current layout matches the specified layout, padding adjustments are applied to the last two dimensions as necessary.

        Example:
            >>> tensor = ttnn.from_torch(torch.rand([1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.clone(tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    )doc";

    bind_registered_operation(
        mod,
        ttnn::clone,
        doc,
        ttnn::nanobind_overload_t{
            [](const std::decay_t<decltype(ttnn::clone)> self,
               const ttnn::Tensor& input,
               const std::optional<ttnn::DataType>& dtype,
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
                return self(input, dtype, memory_config, compute_kernel_config);
            },
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
        });
}
}  // namespace ttnn::operations::data_movement::clone
