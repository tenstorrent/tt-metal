// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>
#include <variant>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "layernorm.hpp"
#include "device/layernorm_device_operation.hpp"
#include "device/layernorm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "device/layernorm_types.hpp"
#include "device/layernorm_common.hpp"

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
    // Bind layernorm_default_compute_config function
    mod.def(
        "layernorm_default_compute_config",
        &ttnn::layernorm_default_compute_config,
        nb::arg("arch"),
        R"doc(
        Returns the default compute kernel config for layernorm.

        Args:
            arch (tt.ARCH): The device architecture.

        Returns:
            ttnn.DeviceComputeKernelConfig: The default compute config for layer norm (HiFi4, approx_mode=False, fp32_dest_acc_en=True).
        )doc");

    // Bind create_layernorm_program_config function
    mod.def(
        "create_layernorm_program_config",
        &ttnn::prim::create_layernorm_program_config,
        nb::arg("shard_spec") = nb::none(),
        nb::arg("tile_height") = 32,
        nb::arg("tile_width") = 32,
        R"doc(
        Creates a program config from shard spec.

        If shard_spec has value, creates a sharded config derived from it.
        Otherwise, returns a default DRAM config.

        Args:
            shard_spec (Optional[tt.ShardSpec]): The shard specification. Defaults to None.
            tile_height (int): The tile height. Defaults to 32.
            tile_width (int): The tile width. Defaults to 32.

        Returns:
            ttnn.LayerNormProgramConfig: The program configuration (either LayerNormDefaultProgramConfig or LayerNormShardedMultiCoreProgramConfig).
        )doc");

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
            - If `residual_input_tensor` is provided, its shape must match the input's logical and padded shape.
            - If TILE: `weight` and `bias` last dim must match input's last dim in both logical and padded shape; their padded height (second-to-last dim) must equal TILE_HEIGHT (i.e. 32).
            - If ROW_MAJOR: `weight` and `bias` last padded dim must be TILE_WIDTH and the stick count must align with the input width.

        )doc";

    ttnn::bind_function<"layer_norm">(
        mod,
        doc,
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
        nb::arg("recip_tensor") = nb::none());
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
            [](ttnn::prim::LayerNormInputs* t, const ttnn::Tensor& input) {
                new (t) ttnn::prim::LayerNormInputs{
                    input, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
            },
            nb::arg("input"))
        .def_rw("input", &ttnn::prim::LayerNormInputs::input)
        .def_rw("residual_input_tensor", &ttnn::prim::LayerNormInputs::residual_input_tensor)
        .def_rw("weight", &ttnn::prim::LayerNormInputs::weight)
        .def_rw("bias", &ttnn::prim::LayerNormInputs::bias)
        .def_rw("stats", &ttnn::prim::LayerNormInputs::stats)
        .def_rw("recip_tensor", &ttnn::prim::LayerNormInputs::recip_tensor);
}

// Fusion-framework façade.
//
// The experimental fusion framework (models/experimental/ops/descriptors) is the sole consumer of
// layernorm's device-op internals.  It used to reach directly into framework symbols
// (LayerNormDeviceOperation::compute_program_hash / select_program_factory,
// <Factory>::create_descriptor / default_core_range).  Those are framework concepts whose shape
// changes under the Metal 2.0 migration (custom program hash removed, create_descriptor becomes
// create_program_artifacts, the pybind-only core_range_set param / default_core_range hook are
// dropped), which would silently break the fusion path.
//
// These free functions are the ONLY Python-facing surface the fusion framework talks to.  They are
// owned by this binding, not by the framework: their NAMES and SIGNATURES are stable, and the
// Metal 2.0 migration only has to re-point their C++ bodies (e.g. wrap a spec hash, lower
// ProgramArtifacts to a ProgramDescriptor).  Python and the fusion framework stay untouched.  Do
// NOT push any of this back into the op's device_operation.hpp — it is a binding-layer concern.
void bind_normalization_layernorm_fusion_facade(nb::module_& mod) {
    mod.def(
        "layer_norm_program_cache_key",
        [](const ttnn::prim::LayerNormParams& operation_attributes, const ttnn::prim::LayerNormInputs& tensor_args) {
            return ttnn::device_operation::detail::compute_program_hash<ttnn::prim::LayerNormDeviceOperation>(
                operation_attributes, tensor_args);
        },
        nb::arg("operation_attributes"),
        nb::arg("tensor_args"),
        R"doc(
        Fusion façade: program-cache key for a layer norm branch.

        Currently the C++ program hash used by the program cache.  Does not run the factory.
        Migration re-points the body (e.g. to a spec hash) without changing this signature.
        )doc");

    mod.def(
        "layer_norm_create_output_tensors",
        &ttnn::prim::LayerNormDeviceOperation::create_output_tensors,
        nb::arg("operation_attributes"),
        nb::arg("tensor_args"),
        R"doc(
        Fusion façade: allocate the output tensor(s) for a layer norm branch.

        For sharded inplace operations this returns the input tensor.

        Args:
            operation_attributes (LayerNormParams): Operation parameters.
            tensor_args (LayerNormInputs): Input tensors.

        Returns:
            ttnn.Tensor: The output tensor for the layer norm operation.
        )doc");

    mod.def(
        "layer_norm_default_core_range",
        &ttnn::prim::LayerNormMultiCoreProgramFactory::default_core_range,
        nb::arg("device"),
        R"doc(
        Fusion façade: default core range for a non-sharded (interleaved) layer norm branch.

        Args:
            device: The device to get the compute grid from.

        Returns:
            ttnn.CoreRangeSet: The default core range covering the device's compute grid.
        )doc");

    mod.def(
        "layer_norm_create_program_descriptor",
        [](const ttnn::prim::LayerNormParams& operation_attributes,
           const ttnn::prim::LayerNormInputs& tensor_args,
           Tensor& tensor_return_value,
           const std::optional<CoreRangeSet>& core_range_set) {
            // Fold factory selection + descriptor creation behind one entry point so the fusion
            // framework never touches the (migration-unstable) factory objects directly.
            auto factory =
                ttnn::prim::LayerNormDeviceOperation::select_program_factory(operation_attributes, tensor_args);
            return std::visit(
                [&](auto&& program_factory) {
                    return program_factory.create_descriptor(
                        operation_attributes, tensor_args, tensor_return_value, core_range_set);
                },
                factory);
        },
        nb::arg("operation_attributes"),
        nb::arg("tensor_args"),
        nb::arg("tensor_return_value"),
        nb::arg("core_range_set") = nb::none(),
        R"doc(
        Fusion façade: build the ProgramDescriptor for a layer norm branch.

        Selects the interleaved vs. sharded factory internally (matching
        LayerNormDeviceOperation::select_program_factory) and returns a runnable ProgramDescriptor.

        Args:
            operation_attributes (LayerNormParams): Operation parameters (norm type, epsilon, memory config, ...).
            tensor_args (LayerNormInputs): Input tensors (input, residual, weight, bias, stats, recip).
            tensor_return_value (ttnn.Tensor): Output tensor reference.
            core_range_set (ttnn.CoreRangeSet, optional): Core range override. For sharded inputs, when
                provided, the shard spec cores are validated to lie within it.

        Returns:
            ttnn.ProgramDescriptor: The program descriptor for the layer norm operation.
        )doc");
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_types(mod);
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
    bind_normalization_layernorm_params_and_inputs(mod);
    bind_normalization_layernorm_fusion_facade(mod);
}

}  // namespace ttnn::operations::normalization::detail

// NOLINTEND(bugprone-unused-raii)
