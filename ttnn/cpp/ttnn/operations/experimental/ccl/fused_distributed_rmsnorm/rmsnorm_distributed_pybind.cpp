// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_distributed_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/ccl/fused_distributed_rmsnorm/rmsnorm_pre_all_gather.hpp"
#include "ttnn/operations/experimental/ccl/fused_distributed_rmsnorm/rmsnorm_post_all_gather.hpp"

namespace ttnn::operations::experimental::ccl {

namespace py = pybind11;

static void bind_rmsnorm_pre_all_gather_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::wan_fused_rmsnorm_pre_allgather,
        R"doc(wan_fused_rmsnorm_pre_allgather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None,
            compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
            memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Computes per-row RMSNorm statistics over the last dimension of :attr:`input_tensor`, producing a
            one-tile-wide tensor that contains sum(x**2) per row (placed in the leftmost column). Intended to be
            followed by an all-gather across devices, then ``wan_fused_rmsnorm_pre_allgather``.

            Notes:
              Supported data types and layouts by tensor ::

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

            Returns:
              - A TILE-layout tensor of dtype ``dtype`` (default BFLOAT16) with the same leading dimensions as
                :attr:`input_tensor` and with the last padded dimension equal to TILE_WIDTH (32). The tile holds
                E(x**2) in its leftmost column for each row.

            Limitations:
              - All tensors must be on device.
              - :attr:`input_tensor` must be BFLOAT16.
              - Inputs must be interleaved memory layout when unsharded.
              - Inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32).
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
        ttnn::experimental::wan_fused_rmsnorm_post_allgather,
        R"doc(wan_fused_rmsnorm_post_allgather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor,
            *, epsilon: float = 1e-5, num_heads_per_device: int = 1,
            weight: Optional[ttnn.Tensor] = None,
            transformation_mat: Optional[ttnn.Tensor] = None,
            rope_cos: Optional[ttnn.Tensor] = None,
            rope_sin: Optional[ttnn.Tensor] = None,
            memory_config: Optional[ttnn.MemoryConfig] = None,
            compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
            dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
            Applies RMSNorm using gathered statistics and optionally fuses per-head output reshape, gamma scaling,
            and rotary positional embeddings (ROPE). This is the second stage of the distributed RMSNorm.

            Notes:
              Supported data types and layouts by tensor ::

              .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

              .. list-table:: stats
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

              .. list-table:: weight (gamma)
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

              .. list-table:: transformation_mat (for ROPE)
                :header-rows: 1

                * - shape
                  - dtype
                  - layout
                * - [1, 1, 32, 32]
                  - BFLOAT16
                  - TILE

              .. list-table:: rope_cos / rope_sin (for ROPE)
                :header-rows: 1

                * - shape
                  - dtype
                  - layout
                * - [1, 1, seq_len, head_dim]
                  - BFLOAT16
                  - TILE

            Arguments:
              - :attr:`epsilon`: Small constant for numerical stability in RMSNorm.
              - :attr:`num_heads_per_device`: Number of attention heads local to each device. The output is reshaped to
                [batch=1, num_heads_per_device, seq_len, hidden_dim/num_heads_per_device].
              - :attr:`weight`: Optional gamma, shape [1, hidden_dim].
              - :attr:`transformation_mat`, :attr:`rope_cos`, :attr:`rope_sin`: Optional tensors enabling ROPE fusion; if
                provided, all three must be present with the shapes/dtypes above.
              - :attr:`dtype`: Optional output dtype override (defaults to input dtype).

            Returns:
              - A TILE-layout tensor with logical shape [1, num_heads_per_device, seq_len, hidden_dim/num_heads_per_device]
                and dtype ``dtype`` if provided, otherwise the input dtype.

            Limitations:
              - All tensors must be on device.
              - :attr:`input_tensor` must be rank-4 with batch dimension 1 and channel dimension 1 (shape [1, 1, S, H]).
              - :attr:`input_tensor`.logical_last_dim must equal its padded last dim; H must be divisible by
                :attr:`num_heads_per_device`.
              - :attr:`stats` last padded dim must be a multiple of TILE_WIDTH (32); its first three padded dims must
                match :attr:`input_tensor`. One tile column per device is expected.
              - If :attr:`weight` is provided it must be TILE layout BFLOAT16 with shape [1, H].
              - When using ROPE, all three tensors (:attr:`transformation_mat`, :attr:`rope_cos`, :attr:`rope_sin`) are required.
              - Inputs cannot be height-sharded; padded height must equal TILE_HEIGHT (32).
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("stats"),
            py::kw_only(),
            py::arg("epsilon") = 1e-5,
            py::arg("num_heads_per_device") = 1,
            py::arg("weight") = std::nullopt,
            py::arg("transformation_mat") = std::nullopt,
            py::arg("rope_cos") = std::nullopt,
            py::arg("rope_sin") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

void py_bind_wan_fused_distributed_rmsnorm(py::module& module) {
    bind_rmsnorm_pre_all_gather_operation(module);
    bind_rmsnorm_post_all_gather_operation(module);
}

}  // namespace ttnn::operations::experimental::ccl
