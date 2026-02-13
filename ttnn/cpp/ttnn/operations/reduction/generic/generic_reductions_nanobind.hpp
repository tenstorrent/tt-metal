// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;

inline std::string get_generic_reduction_doc(const char* op_name, const char* qualified_name) {
    return fmt::format(
        R"doc(
        Computes the {0} of the input tensor :attr:`input_a` along the specified dimension :attr:`dim`.
        If no dimension is provided, {0} is computed over all dimensions yielding a single value.

        Args:
            input_a (ttnn.Tensor): the input tensor. Must be on the device.
            dim (number): dimension value to reduce over.
            keepdim (bool, optional): keep original dimension size. Defaults to `False`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): Compute kernel configuration for the operation. Defaults to `None`.
            scalar (float, optional): A scaling factor to be applied to the input tensor. Defaults to `1.0`.
            correction (bool, optional): Applies only to :func:`ttnn.std` - whether to apply Bessel's correction (i.e. N-1). Defaults to `True`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Subcore grids to use for the operation. Defaults to `None`, which will use all cores.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            The input tensor supports the following data types and layouts:

            .. list-table:: Input Tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - ROW_MAJOR, TILE

            The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width, Height, and ND sharding
            - Output sharding will mirror the input
        )doc",
        op_name,
        qualified_name);
}

inline void bind_generic_reductions(nb::module_& mod) {
    const auto sum_doc = get_generic_reduction_doc("sum", "ttnn.sum");
    ttnn::bind_function<"sum">(
        mod,
        sum_doc.c_str(),
        ttnn::overload_t(
            &ttnn::sum,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));

    const auto mean_doc = get_generic_reduction_doc("mean", "ttnn.mean");
    ttnn::bind_function<"mean">(
        mod,
        mean_doc.c_str(),
        ttnn::overload_t(
            &ttnn::mean,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));

    const auto max_doc = get_generic_reduction_doc("max", "ttnn.max");
    ttnn::bind_function<"max">(
        mod,
        max_doc.c_str(),
        ttnn::overload_t(
            &ttnn::max,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));

    const auto min_doc = get_generic_reduction_doc("min", "ttnn.min");
    ttnn::bind_function<"min">(
        mod,
        min_doc.c_str(),
        ttnn::overload_t(
            &ttnn::min,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));

    const auto std_doc = get_generic_reduction_doc("std", "ttnn.std");
    ttnn::bind_function<"std">(
        mod,
        std_doc.c_str(),
        ttnn::overload_t(
            &ttnn::std,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));

    const auto var_doc = get_generic_reduction_doc("var", "ttnn.var");
    ttnn::bind_function<"var">(
        mod,
        var_doc.c_str(),
        ttnn::overload_t(
            &ttnn::var,
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true,
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::reduction::detail
