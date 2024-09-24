// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "layernorm.hpp"


namespace ttnn::operations::normalization::detail {

namespace py = pybind11;

void bind_normalization_layernorm_program_config(py::module& module) {
    py::class_<LayerNormProgramConfig>(module, "LayerNormProgramConfig").def(py::init<>());

    py::class_<LayerNormDefaultProgramConfig>(module, "LayerNormDefaultProgramConfig")
        .def(py::init<>());

    py::class_<LayerNormShardedMultiCoreProgramConfig>(module, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("subblock_w").noconvert(),
            py::arg("block_h").noconvert(),
            py::arg("block_w").noconvert(),
            py::arg("inplace").noconvert())
        .def(
            "__repr__", [](const LayerNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });
}

void bind_normalization_layernorm_operation(py::module& module) {

    ttnn::bind_registered_operation(
        module,
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
