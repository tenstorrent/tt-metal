// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "layernorm.hpp"
#include "ttnn/operations/experimental/parallel/device/parallel_device_operation_types.hpp"

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::operations::normalization::detail {

struct LayerNormProgramConfigPlaceholder {};

void bind_normalization_layernorm_program_config(nb::module_& mod) {
    nb::class_<LayerNormProgramConfigPlaceholder>(mod, "LayerNormProgramConfig");

    nb::class_<ttnn::prim::LayerNormDefaultProgramConfig>(mod, "LayerNormDefaultProgramConfig")
        .def(
            nb::init<bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("legacy_reduction").noconvert() = false,
            nb::arg("legacy_rsqrt").noconvert() = false,
            nb::arg("use_welford").noconvert() = false)
        .def_rw("legacy_reduction", &prim::LayerNormDefaultProgramConfig::legacy_reduction)
        .def_rw("legacy_rsqrt", &prim::LayerNormDefaultProgramConfig::legacy_rsqrt)
        .def_rw("use_welford", &prim::LayerNormDefaultProgramConfig::use_welford)
        .def("__repr__", [](const ttnn::prim::LayerNormDefaultProgramConfig& config) {
            return fmt::format("{}", config);
        });

    nb::class_<ttnn::prim::LayerNormShardedMultiCoreProgramConfig>(mod, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            nb::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool, bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("subblock_w").noconvert(),
            nb::arg("block_h").noconvert(),
            nb::arg("block_w").noconvert(),
            nb::arg("inplace").noconvert(),
            nb::arg("legacy_reduction").noconvert() = false,
            nb::arg("legacy_rsqrt").noconvert() = false,
            nb::arg("use_welford").noconvert() = false)
        .def_rw(
            "compute_with_storage_grid_size", &prim::LayerNormShardedMultiCoreProgramConfig::compute_with_storage_grid_size)
        .def_rw("subblock_w", &prim::LayerNormShardedMultiCoreProgramConfig::subblock_w)
        .def_rw("block_h", &prim::LayerNormShardedMultiCoreProgramConfig::block_h)
        .def_rw("block_w", &prim::LayerNormShardedMultiCoreProgramConfig::block_w)
        .def_rw("inplace", &prim::LayerNormShardedMultiCoreProgramConfig::inplace)
        .def_rw("legacy_reduction", &prim::LayerNormShardedMultiCoreProgramConfig::legacy_reduction)
        .def_rw("legacy_rsqrt", &prim::LayerNormShardedMultiCoreProgramConfig::legacy_rsqrt)
        .def_rw("use_welford", &prim::LayerNormShardedMultiCoreProgramConfig::use_welford)
        .def("__repr__", [](const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& config) {
            return fmt::format("{}", config);
        });
}

void bind_normalization_layernorm_operation(nb::module_& mod) {
    auto py_operation = ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm,
        R"doc(
        Computes layer norm over :attr:`input_tensor`.
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

            .. list-table:: output_tensor
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            Output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width and Block sharded

        Limitations:
            - All input tensors must be on-device and have a rank >= 1.
            - Unsharded tensors must be interleaved, sharded tensors cannot be height sharded.
            - If the input is sharded, the :attr:`output` and :attr:`residual_input_tensor` must have identical shard spec and memory config.
            - If `residual_input_tensor` is provided, it must match the input's padded shape.
            - If TILE: `weight` and `bias` padded dim must match input's last padded dim; padded height must equal TILE_HEIGHT (i.e. 32).
            - If ROW_MAJOR: `weight` and `bias` last padded dim must be TILE_WIDTH and the stick count must align with the input width.

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

    // Add branch() method for parallel execution support
    // Returns BranchDescriptor by value (move semantics)
    py_operation.def(
        "branch",
        [](const std::decay_t<decltype(ttnn::layer_norm)>& /*self*/,
           const ttnn::Tensor& input_tensor,
           const tt::tt_metal::CoreRangeSet& cores,
           float epsilon,
           const std::optional<const ttnn::Tensor>& weight,
           const std::optional<const ttnn::Tensor>& bias,
           const std::optional<const ttnn::Tensor>& residual_input_tensor,
           const std::optional<MemoryConfig>& memory_config,
           const std::optional<const prim::LayerNormProgramConfig>& program_config,
           std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
            return ExecuteLayerNorm::branch(
                input_tensor,
                cores,
                epsilon,
                weight,
                bias,
                residual_input_tensor,
                memory_config,
                program_config,
                compute_kernel_config);
        },
        nb::arg("input_tensor"),
        nb::arg("cores"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12f,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
            Create a branch descriptor for parallel execution with ttnn.parallel.

            This allows running multiple Layer norm operations in parallel on disjoint
            core ranges within a single fused program.

            Args:
                input_tensor (ttnn.Tensor): Input tensor to normalize.
                cores (ttnn.CoreRangeSet): Core range for this branch (must be disjoint from other branches).

            Keyword Args:
                epsilon (float): Small constant for numerical stability. Defaults to 1e-12.
                weight (ttnn.Tensor, optional): Gamma scale tensor. Defaults to None.
                bias (ttnn.Tensor, optional): Beta bias tensor. Defaults to None.
                residual_input_tensor (ttnn.Tensor, optional): Residual tensor. Defaults to None.
                memory_config (ttnn.MemoryConfig, optional): Output memory config. Defaults to None.
                program_config (ttnn.ProgramConfig, optional): Program config. Defaults to None.
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute config. Defaults to None.

            Returns:
                BranchDescriptor: A branch descriptor for use with ttnn.parallel().

            Example:
                >>> cores_a = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> cores_b = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
                >>> branch_a = ttnn.layer_norm.branch(input_a, cores_a, epsilon=1e-5, weight=w_a, bias=b_a)
                >>> branch_b = ttnn.layer_norm.branch(input_b, cores_b, epsilon=1e-5, weight=w_b, bias=b_b)
                >>> results = ttnn.parallel([branch_a, branch_b])
        )doc");
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail

// NOLINTEND(bugprone-unused-raii)
