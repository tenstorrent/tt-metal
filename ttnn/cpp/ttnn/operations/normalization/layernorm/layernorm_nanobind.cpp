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
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "layernorm.hpp"
#include "device/layernorm_op_multi_core.hpp"
#include "device/layernorm_op_multi_core_sharded.hpp"
#include "device/layernorm_device_operation.hpp"
#include "device/layernorm_device_operation_types.hpp"
#include "device/layernorm_types.hpp"

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::operations::normalization::detail {

struct LayerNormProgramConfigPlaceholder {};

void bind_normalization_layernorm_types(nb::module_& mod) {
    export_enum<ttnn::prim::LayerNormType>(mod, "LayerNormType");
    export_enum<ttnn::prim::DistributedLayerNormStage>(mod, "DistributedLayerNormStage");
}

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
    const auto* doc = R"doc(
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

        )doc";

    ttnn::bind_function<"layer_norm">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                float,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<const ttnn::Tensor>&,
                const std::optional<ttnn::MemoryConfig>&,
                const std::optional<const ttnn::prim::LayerNormProgramConfig>&,
                std::optional<const ttnn::DeviceComputeKernelConfig>,
                const std::optional<const ttnn::Tensor>&>(&ttnn::layer_norm),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("recip_tensor") = nb::none()));
}

void bind_normalization_layernorm_params_and_inputs(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormParams>(mod, "LayerNormParams")
        .def(nb::init<>())
        .def_rw("norm_type", &ttnn::prim::LayerNormParams::norm_type)
        .def_rw("distributed_norm_stage", &ttnn::prim::LayerNormParams::distributed_norm_stage)
        .def_rw("eps", &ttnn::prim::LayerNormParams::eps)
        .def_rw("output_mem_config", &ttnn::prim::LayerNormParams::output_mem_config)
        .def_rw("program_config", &ttnn::prim::LayerNormParams::program_config)
        .def_rw("compute_kernel_config", &ttnn::prim::LayerNormParams::compute_kernel_config)
        .def_rw("dtype", &ttnn::prim::LayerNormParams::dtype);

    nb::class_<ttnn::prim::LayerNormInputs>(mod, "LayerNormInputs")
        .def(
            "__init__",
            [](ttnn::prim::LayerNormInputs* t) {
                new (t) ttnn::prim::LayerNormInputs{
                    ttnn::Tensor(), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
            })
        .def_rw("input", &ttnn::prim::LayerNormInputs::input)
        .def_rw("residual_input_tensor", &ttnn::prim::LayerNormInputs::residual_input_tensor)
        .def_rw("weight", &ttnn::prim::LayerNormInputs::weight)
        .def_rw("bias", &ttnn::prim::LayerNormInputs::bias)
        .def_rw("stats", &ttnn::prim::LayerNormInputs::stats)
        .def_rw("recip_tensor", &ttnn::prim::LayerNormInputs::recip_tensor);
}

void bind_normalization_layernorm_device_operation(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormDeviceOperation>(mod, "LayerNormDeviceOperation")
        .def_static(
            "create_output_tensors",
            &ttnn::prim::LayerNormDeviceOperation::create_output_tensors,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            R"doc(
            Creates output tensors for the layer norm operation.

            This creates appropriately configured output tensors based on the operation
            attributes and input tensors. For sharded operations with inplace=True,
            returns the input tensor.

            Args:
                operation_attributes (LayerNormParams): Operation parameters.
                tensor_args (LayerNormInputs): Input tensors.

            Returns:
                ttnn.Tensor: The output tensor for the layer norm operation.
            )doc")
        .def_static(
            "compute_output_specs",
            &ttnn::prim::LayerNormDeviceOperation::compute_output_specs,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            R"doc(
            Computes the output tensor specification for the layer norm operation.

            Args:
                operation_attributes (LayerNormParams): Operation parameters.
                tensor_args (LayerNormInputs): Input tensors.

            Returns:
                ttnn.TensorSpec: The output tensor specification.
            )doc")
        .def_static(
            "select_program_factory",
            &ttnn::prim::LayerNormDeviceOperation::select_program_factory,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            R"doc(
            Selects the appropriate program factory based on input tensor's memory layout.

            Returns LayerNormShardedProgramFactory for sharded inputs,
            LayerNormMultiCoreProgramFactory for non-sharded inputs.

            Args:
                operation_attributes (LayerNormParams): Operation parameters.
                tensor_args (LayerNormInputs): Input tensors.

            Returns:
                Union[LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory]:
                    The appropriate program factory for the input tensor.
            )doc");
}

void bind_normalization_layernorm_program_factory(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormMultiCoreProgramFactory>(mod, "LayerNormMultiCoreProgramFactory")
        .def_static(
            "create_descriptor",
            [](const ttnn::prim::LayerNormParams& operation_attributes,
               const ttnn::prim::LayerNormInputs& tensor_args,
               Tensor& tensor_return_value,
               const std::optional<CoreRangeSet>& core_range_set) {
                return ttnn::prim::LayerNormMultiCoreProgramFactory::create_descriptor(
                    operation_attributes, tensor_args, tensor_return_value, core_range_set);
            },
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            nb::arg("tensor_return_value"),
            nb::arg("core_range_set") = std::nullopt,
            R"doc(
            Creates a program descriptor for layer norm multi-core operation.

            Args:
                operation_attributes (LayerNormParams): Operation parameters including norm type, epsilon, memory config, etc.
                tensor_args (LayerNormInputs): Input tensors including input, residual, weight, bias, and stats.
                tensor_return_value (ttnn.Tensor): Output tensor reference.
                core_range_set (ttnn.CoreRangeSet, optional): Optional core range set to restrict the program to specific cores.
                    If not provided, uses device's compute grid.

            Returns:
                ttnn.ProgramDescriptor: The program descriptor for the layer norm operation.
            )doc");

    nb::class_<ttnn::prim::LayerNormShardedProgramFactory>(mod, "LayerNormShardedProgramFactory")
        .def_static(
            "create_descriptor",
            [](const ttnn::prim::LayerNormParams& operation_attributes,
               const ttnn::prim::LayerNormInputs& tensor_args,
               Tensor& tensor_return_value,
               const std::optional<CoreRangeSet>& core_range_set) {
                return ttnn::prim::LayerNormShardedProgramFactory::create_descriptor(
                    operation_attributes, tensor_args, tensor_return_value, core_range_set);
            },
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            nb::arg("tensor_return_value"),
            nb::arg("core_range_set") = std::nullopt,
            R"doc(
            Creates a program descriptor for sharded layer norm operation.

            Args:
                operation_attributes (LayerNormParams): Operation parameters including norm type, epsilon, memory config, etc.
                    Must have a LayerNormShardedMultiCoreProgramConfig as the program_config.
                tensor_args (LayerNormInputs): Input tensors including input (sharded), residual, weight, bias, and stats.
                tensor_return_value (ttnn.Tensor): Output tensor reference (sharded).
                core_range_set (ttnn.CoreRangeSet, optional): Optional core range set. If provided, validates that the
                    sharded tensor's shard spec cores lie entirely within this core range set. Raises an error if any
                    shard spec core is outside the provided range.

            Returns:
                ttnn.ProgramDescriptor: The program descriptor for the sharded layer norm operation.

            Raises:
                RuntimeError: If core_range_set is provided and the sharded tensor's shard spec cores are not
                    entirely contained within it.
            )doc");
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_types(mod);
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
    bind_normalization_layernorm_params_and_inputs(mod);
    bind_normalization_layernorm_device_operation(mod);
    bind_normalization_layernorm_program_factory(mod);
}

}  // namespace ttnn::operations::normalization::detail

// NOLINTEND(bugprone-unused-raii)
