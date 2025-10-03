// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_distributed_pybind.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "layernorm_pre_all_gather.hpp"
#include "layernorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_pre_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm_pre_all_gather,
        R"doc(
        Computes statistics for distributed layer normalization.

        This operation computes sum(input_tensor^2) and sum(input_tensor) over the last dimension.
        These statistics are then used in a distributed setting where they are gathered across devices
        before completing the layer normalization in the post all-gather step.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to compute statistics for. Must be rank 4 and on-device.

        Keyword Args:
            dtype (Optional[ttnn.DataType]): Output data type. Defaults to BFLOAT16.
            residual_input_tensor (Optional[ttnn.Tensor]): Optional residual tensor to add. Defaults to None.
            compute_kernel_config (Optional): Compute kernel configuration. Defaults to None.
            program_config (Optional): Program configuration. Defaults to None.
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to None.

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
            - Input tensors must be on-device and rank 4.
            - Unsharded runs: input_tensor must be interleaved.
            - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32).
            - When using residual_input_tensor with sharding, it must match the input_tensor padded shape and sharding.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("residual_input_tensor") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

void bind_normalization_layernorm_post_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm_post_all_gather,
        R"doc(
        Completes distributed layer normalization using gathered statistics.

        Performs the second part of a distributed layer normalization operation, normalizing the input
        based on the gathered statistics from all devices. This is typically used after an all-gather
        operation has combined the statistics from layer_norm_pre_all_gather across devices.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to normalize. Must be rank 4 and on-device.
            stats (ttnn.Tensor): Gathered statistics tensor from pre all-gather step.

        Keyword Args:
            epsilon (float): Small value to prevent division by zero. Defaults to 1e-12.
            weight (Optional[ttnn.Tensor]): Gamma parameter for affine transformation. Defaults to None.
            bias (Optional[ttnn.Tensor]): Beta parameter for affine transformation. Defaults to None.
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to None.
            compute_kernel_config (Optional): Compute kernel configuration. Defaults to None.
            program_config (Optional): Program configuration. Defaults to None.
            dtype (Optional[ttnn.DataType]): Output data type. Defaults to None (same as input).

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
                * - BFLOAT16
                  - ROW_MAJOR

        Limitations:
            - Input tensors must be on-device and rank 4.
            - The last padded dim of stats must be a multiple of TILE_WIDTH.
            - The first three padded dims of stats must match input_tensor.
            - If weight (gamma) is provided, bias (beta) must also be provided with matching layouts with their last padded dim matching TILE_WIDTH.
            - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32), and stats must be sharded with num_cores=1 and expected tile columns per device.
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
            py::arg("dtype") = std::nullopt});
}

void bind_normalization_layernorm_distributed(py::module& module) {
    bind_normalization_layernorm_pre_all_gather_operation(module);
    bind_normalization_layernorm_post_all_gather_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
