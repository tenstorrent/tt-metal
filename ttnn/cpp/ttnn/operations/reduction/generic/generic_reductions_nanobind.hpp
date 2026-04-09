// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;

inline std::string get_generic_reduction_doc(const char* op_name, const char* qualified_name) {
    return fmt::format(
        R"doc(
        Computes the {0} of the input tensor :attr:`input_a` along the specified dimension(s) :attr:`dim`.
        If no dimension is provided, {0} is computed over all dimensions yielding a single value.

        Args:
            input_a (ttnn.Tensor): the input tensor. Must be on the device.
            dim (number or tuple): dimension value(s) to reduce over.
            keepdim (bool, optional): keep the original dimension size(s). Defaults to `False`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): Compute kernel configuration for the operation. Defaults to `None`.
            scalar (float, optional): A scaling factor to be applied to the input tensor. Defaults to `1.0`.
            correction (bool, optional): **Deprecated.** This parameter is deprecated and will be removed in a future release. It has no impact on the result.
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
                  - TILE

            The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width, Height, and ND sharding
            - Output sharding will mirror the input
        )doc",
        op_name,
        qualified_name);
}

// Wrapper that detects explicit use of the deprecated 'correction' parameter and
// emits a Python DeprecationWarning before forwarding to the real implementation.
// The binding-layer type is std::optional<bool> (default nb::none()) so we can
// distinguish "user passed correction=True" from "used the default".
template <auto Func>
Tensor generic_reduction_with_deprecated_correction(
    const Tensor& input_tensor,
    const std::optional<std::variant<int, int64_t, SmallVector<int>>>& dim,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    std::optional<bool> correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (correction.has_value()) {
        PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "The 'correction' parameter is deprecated and will be removed in a future release.",
            1);
    }
    return Func(
        input_tensor,
        dim,
        keepdim,
        memory_config,
        compute_kernel_config,
        scalar,
        correction.value_or(true),
        sub_core_grids);
}

inline void bind_generic_reductions(nb::module_& mod) {
    const auto sum_doc = get_generic_reduction_doc("sum", "ttnn.sum");
    ttnn::bind_function<"sum">(
        mod,
        sum_doc.c_str(),
        &generic_reduction_with_deprecated_correction<&ttnn::sum>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());

    const auto mean_doc = get_generic_reduction_doc("mean", "ttnn.mean");
    ttnn::bind_function<"mean">(
        mod,
        mean_doc.c_str(),
        &generic_reduction_with_deprecated_correction<&ttnn::mean>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());

    const auto max_doc = get_generic_reduction_doc("max", "ttnn.max");
    ttnn::bind_function<"max">(
        mod,
        max_doc.c_str(),
        &generic_reduction_with_deprecated_correction<&ttnn::max>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());

    const auto min_doc = get_generic_reduction_doc("min", "ttnn.min");
    ttnn::bind_function<"min">(
        mod,
        min_doc.c_str(),
        &generic_reduction_with_deprecated_correction<&ttnn::min>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::reduction::detail
