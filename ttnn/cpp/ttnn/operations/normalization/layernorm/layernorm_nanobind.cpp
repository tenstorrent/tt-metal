// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "layernorm.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_layernorm_program_config(nb::module_& mod) {
    nb::class_<LayerNormProgramConfig>(mod, "LayerNormProgramConfig").def(nb::init<>());

    nb::class_<LayerNormDefaultProgramConfig>(mod, "LayerNormDefaultProgramConfig")
        .def(
            nb::init<bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("legacy_reduction").noconvert() = true,
            nb::arg("legacy_rsqrt").noconvert() = true,
            nb::arg("use_welford").noconvert() = false)
        .def("__repr__", [](const LayerNormDefaultProgramConfig& config) { return fmt::format("{}", config); });

    nb::class_<LayerNormShardedMultiCoreProgramConfig>(mod, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            nb::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("subblock_w").noconvert(),
            nb::arg("block_h").noconvert(),
            nb::arg("block_w").noconvert(),
            nb::arg("inplace").noconvert(),
            nb::arg("legacy_reduction").noconvert() = true,
            nb::arg("legacy_rsqrt").noconvert() = true)
        .def(
            "__repr__", [](const LayerNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });
}

void bind_normalization_layernorm_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
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

        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail
