// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_distributed_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "rmsnorm_pre_all_gather.hpp"
#include "rmsnorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_rmsnorm_pre_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::rms_norm_pre_all_gather,
        R"doc(
        Computes statistics for distributed RMS normalization.

        This operation computes sum(input_tensor^2) and sum(input_tensor) over the last dimension.
        These statistics are then used in a distributed setting where they are gathered across devices
        before completing the RMS normalization in the post all-gather step.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to compute statistics for. Must be on-device.

        Keyword Args:
            dtype (Optional[ttnn.DataType]): Output data type. Defaults to BFLOAT16.
            residual_input_tensor (Optional[ttnn.Tensor]): Optional residual tensor to add. Defaults to None.
            compute_kernel_config (Optional): Compute kernel configuration. Defaults to None.
            program_config (Optional): Program configuration. Defaults to None.
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to None.
            use_2d_core_grid (Optional[bool]): Whether to use 2D core grid. Defaults to None.

        Returns:
            ttnn.Tensor: Statistics tensor containing sum and sum of squares.

        Note:
            Supported data types and layouts:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

            .. list-table:: residual_input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

        Limitations:
            - All tensors must be on-device.
            - Unsharded inputs must be interleaved.
            - Sharded inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32). If residual_input_tensor is provided, it must match input's padded shape and sharding.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("residual_input_tensor") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("use_2d_core_grid") = std::nullopt});
}

void bind_normalization_rmsnorm_post_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::rms_norm_post_all_gather,
        R"doc(
        Completes distributed RMS normalization using gathered statistics.

        Performs the second part of a distributed RMS normalization operation, normalizing the input
        based on the gathered statistics from all devices. This is typically used after an all-gather
        operation has combined the statistics from rms_norm_pre_all_gather across devices.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to normalize. Must be on-device.
            stats (ttnn.Tensor): Gathered statistics tensor from pre all-gather step.

        Keyword Args:
            epsilon (float): Small value to prevent division by zero. Defaults to 1e-12.
            weight (Optional[ttnn.Tensor]): Gamma parameter for scaling. Defaults to None.
            bias (Optional[ttnn.Tensor]): Beta parameter for shifting. Defaults to None.
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to None.
            compute_kernel_config (Optional): Compute kernel configuration. Defaults to None.
            program_config (Optional): Program configuration. Defaults to None.
            dtype (Optional[ttnn.DataType]): Output data type. Defaults to None (same as input).
            use_2d_core_grid (Optional[bool]): Whether to use 2D core grid. Defaults to None.

        Returns:
            ttnn.Tensor: Normalized output tensor.

        Note:
            Supported data types and layouts:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE

            .. list-table:: stats
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, BFLOAT8_B
                  - TILE

            .. list-table:: weight (gamma) and bias (beta)
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32
                  - TILE, ROW_MAJOR

        Limitations:
            - All tensors must be on-device.
            - The last padded dim of stats must be a multiple of TILE_WIDTH, and its first three padded dims must match input_tensor.
            - If weight (gamma) is provided, bias (beta) must also be provided. Gamma and beta must have the same layout. If this is ROW_MAJOR, last padded dim must be TILE_WIDTH.
            - Sharded runs: inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32). When sharded, stats must be sharded across one core.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("stats"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("use_2d_core_grid") = std::nullopt});
}

void bind_normalization_rms_norm_distributed(py::module& module) {
    bind_normalization_rmsnorm_pre_all_gather_operation(module);
    bind_normalization_rmsnorm_post_all_gather_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
