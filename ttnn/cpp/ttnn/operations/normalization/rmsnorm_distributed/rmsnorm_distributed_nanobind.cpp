// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_distributed_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "rmsnorm_pre_all_gather.hpp"
#include "rmsnorm_post_all_gather.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_rmsnorm_pre_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("use_2d_core_grid") = nb::none()});
}

void bind_normalization_rmsnorm_post_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("stats"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("use_2d_core_grid") = nb::none()});
}

void bind_normalization_rms_norm_distributed(nb::module_& mod) {
    bind_normalization_rmsnorm_pre_all_gather_operation(mod);
    bind_normalization_rmsnorm_post_all_gather_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail
