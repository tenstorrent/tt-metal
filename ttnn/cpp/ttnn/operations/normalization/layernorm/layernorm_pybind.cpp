// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pybind.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "layernorm.hpp"

namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_program_config(py::module& module) {
    py::class_<LayerNormProgramConfig>(module, "LayerNormProgramConfig").def(py::init<>());

    py::class_<LayerNormDefaultProgramConfig>(module, "LayerNormDefaultProgramConfig")
        .def(
            py::init<bool, bool, bool>(),
            py::kw_only(),
            py::arg("legacy_reduction").noconvert() = true,
            py::arg("legacy_rsqrt").noconvert() = true,
            py::arg("use_welford").noconvert() = false)
        .def("__repr__", [](const LayerNormDefaultProgramConfig& config) { return fmt::format("{}", config); });

    py::class_<LayerNormShardedMultiCoreProgramConfig>(module, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool, bool, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert(),
            py::arg("inplace").noconvert(),
            py::arg("legacy_reduction").noconvert() = true,
            py::arg("legacy_rsqrt").noconvert() = true)
        .def(
            "__repr__", [](const LayerNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });
}

void bind_normalization_layernorm_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm,
        R"doc(
            Compute layer norm over :attr:`input_tensor`.
            See `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ for more details.

            .. math::

                \text{layer_norm}(x, \gamma, \beta, \epsilon) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta

            Where:
                - :math:`\mu` is the mean of the input tensor. This is computed over the last dimension of the input tensor (W).
                - :math:`\sigma^2` is the variance of the input tensor. This is computed over the last dimension of the input tensor (W) and is biased.
                - :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively
                - :math:`\epsilon` is a small constant


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            epsilon (float): 1e-12.
            weight (ttnn.Tensor, optional): Defaults to `None`.
            bias (ttnn.Tensor, optional): Defaults to `None`.
            residual_input_tensor (ttnn.Tensor, optional): Defaults to `None`.
            program_config (ttnn.ProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig)

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported data types and layouts by tensor:

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

            .. list-table:: weight (gamma) and bias (beta)
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

            .. list-table:: stats (POST_ALL_GATHER only)
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16
                 - TILE

            .. list-table:: output_tensor
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32, BFLOAT8_B (typically matches input; PRE_ALL_GATHER produces BF16)
                 - TILE

        Limitations:
            - All input tensors must be on-device and have a rank >= 1.
            - Unsharded tensors must be interleaved, sharded tensors cannot be height sharded.
            - If `residual_input_tensor` is provided, it must match the input's padded shape.
            - If TILE: `weight` and `bias` padded dim must match input's last padded dim; padded height must equal TILE_HEIGHT (i.e. 32).
            - If ROW_MAJOR: `weight` and `bias` last padded dim must be TILE_WIDTH and the stick count must align with the input width.
            - If the input is sharded, the :attr:`output` and :attr:`residual_input_tensor` must have identical shard spec and memory config.

        Example:
            .. code-block:: python

                input_tensor = ttnn.rand([32, 64], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
                output_tensor = ttnn.layer_norm(input_tensor)

        )doc",

        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("residual_input_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

void bind_normalization_layernorm(py::module& module) {
    bind_normalization_layernorm_program_config(module);
    bind_normalization_layernorm_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
