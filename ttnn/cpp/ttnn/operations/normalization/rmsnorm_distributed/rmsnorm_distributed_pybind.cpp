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
        R"doc(rms_norm_pre_all_gather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
            Compute sum(:attr:`input_tensor`ˆ2) and sum(:attr:`input_tensor`) over the last dimension.

            Note:
              Supported data types and layouts by tensor ::

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
              - Unsharded inputs must be interleaved
              - Sharded inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32). If :attr:`residual_input_tensor` is provided, it must match input's padded shape and sharding.

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
        R"doc(rms_norm_post_all_gather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Performs the second part of a distributed RMSNorm operation normalizing the input based on the gathered statistics input.

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
              - The last padded dim of :attr:`stats` must be a multiple of TILE_WIDTH, and its first three padded dims must match :attr:`input_tensor`.
              - If :attr:`weight` (gamma) is provided, :attr:`bias` (beta) must also be provided. Gamma and beta must have the same layout. If this is ROW_MAJOR, last padded dim must be TILE_WIDTH.
              - Sharded runs: inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32). When sharded, :attr:`stats` must be sharded across one core.
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
