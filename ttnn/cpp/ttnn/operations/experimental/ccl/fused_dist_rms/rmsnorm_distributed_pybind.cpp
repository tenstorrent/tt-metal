// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_distributed_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/ccl/fused_dist_rms/rmsnorm_pre_all_gather.hpp"
#include "ttnn/operations/experimental/ccl/fused_dist_rms/rmsnorm_post_all_gather.hpp"

namespace ttnn::operations::experimental::ccl {

namespace py = pybind11;

static void bind_rmsnorm_pre_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::fused_rmsnorm_pre_allgather,
        R"doc(fused_rmsnorm_pre_allgather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
            Compute sum(:attr:`input_tensor`ˆ2) and sum(:attr:`input_tensor`) over the last dimension.

            Note:
              Supported data types and layouts by tensor ::

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32, BFLOAT8_B
                  - TILE

            Limitations:
              - All tensors must be on-device.
              - Unsharded inputs must be interleaved
              - Sharded inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32).

              )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

static void bind_rmsnorm_post_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::fused_rmsnorm_post_allgather,
        R"doc(fused_rmsnorm_post_allgather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor, epsilon: float = 1e-5, num_heads_per_device: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
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

            .. list-table:: weight (gamma)
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16, FLOAT32
                  - TILE, ROW_MAJOR

            Limitations:
              - All tensors must be on-device.
              - The last padded dim of :attr:`stats` must be a multiple of TILE_WIDTH, and its first three padded dims must match :attr:`input_tensor`.
              - If :attr:`weight` (gamma) is provided it must be ROW_MAJOR or TILE. If ROW_MAJOR, last padded dim must be TILE_WIDTH.
              - Sharded runs: inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32). When sharded, :attr:`stats` must be sharded across one core.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("stats"),
            py::kw_only(),
            py::arg("epsilon") = 1e-5,
            py::arg("num_heads_per_device") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

void py_bind_fused_dist_rmsnorm(py::module& module) {
    bind_rmsnorm_pre_all_gather_operation(module);
    bind_rmsnorm_post_all_gather_operation(module);
}

}  // namespace ttnn::operations::experimental::ccl
