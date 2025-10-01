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
        R"doc(layer_norm_pre_all_gather(input_tensor: ttnn.Tensor, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor
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
              - Input tensors must be on-device and rank 4.
              - Unsharded runs: :attr:`input_tensor` must be interleaved.
              - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32).
              - When using :attr:`residual_input_tensor` with sharding, it must match the :attr:`input_tensor` padded shape and sharding.
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
        R"doc(layer_norm_post_all_gather(input_tensor: ttnn.Tensor, stats: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Performs the second part of a distributed layernorm operation normalizing the input based on the gathered statistics input.

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
              - The last padded dim of :attr:`stats` must be a multiple of TILE_WIDTH.
              - The first three padded dims of :attr:`stats` must match :attr:`input_tensor`.
              - If :attr:`weight` (gamma) is provided, :attr:`bias` (beta) must also be provided with matching layouts with their last padded dim matching TILE_WIDTH.
              - Sharded runs: inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32), and :attr:`stats` must be sharded with `num_cores=1` and expected tile columns per device.
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
