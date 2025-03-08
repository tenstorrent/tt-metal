// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "groupnorm.hpp"

namespace ttnn::operations::normalization::detail {
namespace py = pybind11;

void bind_normalization_groupnorm_program_config(py::module& module) {
    py::class_<GroupNormProgramConfig>(module, "GroupNormProgramConfig").def(py::init<>());

    py::class_<GroupNormMultiCoreProgramConfig>(module, "GroupNormMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, MathFidelity, DataType, DataType, bool, Layout>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("math_fidelity"),
            py::arg("im_data_format"),
            py::arg("out_data_format"),
            py::arg("inplace").noconvert(),
            py::arg("output_layout"))
        .def("__repr__", [](const GroupNormMultiCoreProgramConfig& config) { return fmt::format("{}", config); });

    py::class_<GroupNormShardedMultiCoreProgramConfig>(module, "GroupNormShardedMultiCoreProgramConfig")
        .def(
            py::init<CoreCoord, MathFidelity, DataType, DataType, bool, Layout>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("math_fidelity"),
            py::arg("im_data_format"),
            py::arg("out_data_format"),
            py::arg("inplace").noconvert(),
            py::arg("output_layout"))
        .def(
            "__repr__", [](const GroupNormShardedMultiCoreProgramConfig& config) { return fmt::format("{}", config); });
}

void bind_normalization_groupnorm_operation(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::group_norm,
        R"doc(
            Compute group_norm over :attr:`input_tensor`.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            num_groups (int)
            epsilon (float): 1e-12.
            input_mask (ttnn.Tensor, optional): Defaults to `None`.
            weight (ttnn.Tensor, optional): Defaults to `None`.
            bias (ttnn.Tensor, optional): Defaults to `None`.
            dtype (ttnn.DataType, optional): Defaults to `None`.
            core_grid (CoreGrid, optional): Defaults to `None`.
            inplace (bool, optional): Defaults to `True`.
            output_layout (ttnn.Layout, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("num_groups"),
            py::arg("epsilon") = 1e-12,
            py::arg("input_mask") = std::nullopt,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("inplace") = true,
            py::arg("output_layout") = std::nullopt});
}
void bind_normalization_groupnorm(py::module& module) {
    bind_normalization_groupnorm_program_config(module);
    bind_normalization_groupnorm_operation(module);
}

}  // namespace ttnn::operations::normalization::detail
