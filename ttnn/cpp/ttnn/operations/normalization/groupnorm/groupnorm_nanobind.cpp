// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "groupnorm.hpp"

namespace nb = nanobind;

namespace ttnn::operations::normalization::detail {

namespace {
void bind_normalization_group_norm_operation(nb::module_& mod) {

    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("num_groups"),
            nb::arg("epsilon") = 1e-12,
            nb::arg("input_mask") = std::nullopt,
            nb::arg("weight") = std::nullopt,
            nb::arg("bias") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("core_grid") = std::nullopt,
            nb::arg("inplace") = true,
            nb::arg("output_layout") = std::nullopt,
            nb::arg("num_out_blocks") = std::nullopt});
}
} // namespace

void bind_normalization_group_norm(nb::module_& mod) { bind_normalization_group_norm_operation(mod); }

}  // namespace ttnn::operations::normalization::detail
