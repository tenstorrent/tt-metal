// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "layernorm.hpp"

namespace nb = nanobind;

namespace ttnn::operations::normalization::detail {

void bind_normalization_layernorm_program_config(nb::module_& mod) {
    nb::class_<LayerNormProgramConfig>(mod, "LayerNormProgramConfig")
        .def(nb::init<>());

    nb::class_<LayerNormDefaultProgramConfig>(mod, "LayerNormDefaultProgramConfig")
        .def(nb::init<>());

    nb::class_<LayerNormShardedMultiCoreProgramConfig>(mod, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            nb::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("subblock_w").noconvert(),
            nb::arg("block_h").noconvert(),
            nb::arg("block_w").noconvert(),
            nb::arg("inplace").noconvert())
        .def(
            "__repr__", [](const LayerNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });
}

void bind_normalization_layernorm_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::layer_norm,
        R"doc(
            Compute layer_norm over :attr:`input_tensor`.


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


        )doc",

        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = std::nullopt,
            nb::arg("bias") = std::nullopt,
            nb::arg("residual_input_tensor") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("program_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
}

}  // namespace ttnn::operations::normalization::detail
