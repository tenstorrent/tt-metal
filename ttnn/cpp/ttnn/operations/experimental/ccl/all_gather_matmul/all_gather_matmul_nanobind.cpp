// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

// TODO: Update with all_gather_matmul docs
void bind_all_gather_matmul(nb::module_& mod) {

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::all_gather_matmul,
        R"doc(all_gather_matmul(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.all_gather_matmul(tensor, weight_tensor, dim=0, (0, 0))

        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("dim"),
            nb::arg("all_gather_core_grid_offset"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_ag") = std::nullopt,
            nb::arg("num_workers") = std::nullopt,
            nb::arg("num_buffers_per_channel") = std::nullopt,
            nb::arg("memory_config_mm") = std::nullopt,
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("dtype") = std::nullopt,
            nb::arg("program_config") = std::nullopt,
            nb::arg("activation") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("core_grid") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::ccl
