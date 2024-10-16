// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "groupnorm.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_group_norm(pybind11::module& module) {
    namespace py = pybind11;

    ttnn::bind_registered_operation(module,
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
                                    ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                                             py::kw_only(),
                                                             py::arg("num_groups"),
                                                             py::arg("epsilon") = 1e-12,
                                                             py::arg("input_mask") = std::nullopt,
                                                             py::arg("weight") = std::nullopt,
                                                             py::arg("bias") = std::nullopt,
                                                             py::arg("memory_config") = std::nullopt,
                                                             py::arg("dtype") = std::nullopt,
                                                             py::arg("core_grid") = std::nullopt,
                                                             py::arg("inplace") = true,
                                                             py::arg("output_layout") = std::nullopt});
}

}  // namespace ttnn::operations::normalization::detail
