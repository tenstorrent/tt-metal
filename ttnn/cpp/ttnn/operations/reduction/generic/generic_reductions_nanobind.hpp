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

inline std::string get_generic_reduction_doc(
    const char* op_name, const char* qualified_name, bool int32_supported, bool has_fast_approximate_mode = false) {
    // INT32 (TILE only) is supported for sum/min/max via the SFPU reduce path, but not for mean:
    // the average (sum / N) is usually fractional, so there is no canonical INT32 result to return.
    const char* int32_row = int32_supported ? R"doc(
                * - INT32
                  - TILE)doc"
                                            : "";
    // Only ttnn.mean exposes fast_and_approximate_mode.
    const char* fast_approx_kwarg = has_fast_approximate_mode ? R"doc(
            fast_and_approximate_mode (bool, optional): FLOAT32 only. `False` (default) uses the accurate SFPU path (full float32 accumulation); `True` uses the faster FPU path (inputs truncated to TF32, higher ULP error). The accurate path requires a compute_kernel_config with `fp32_dest_acc_en=True` and is unavailable on Quasar; in those cases it falls back to the FPU. No effect for non-FLOAT32 inputs.)doc"
                                                              : "";
    return fmt::format(
        R"doc(
        Computes the {0} of the input tensor :attr:`input_tensor` along the specified dimension(s) :attr:`dim`.
        If no dimension is provided, {0} is computed over all dimensions yielding a single value.

        Args:
            input_tensor (ttnn.Tensor): the input tensor. Must be on the device.
            dim (number or tuple): dimension value(s) to reduce over.
            keepdim (bool, optional): keep the original dimension size(s). Defaults to `False`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): Compute kernel configuration for the operation. Defaults to `None`.
            scalar (float, optional): A scaling factor to be applied to the input tensor. Defaults to `1.0`.
            correction (bool, optional): **Deprecated.** This parameter is deprecated and will be removed in a future release. It has no impact on the result.
            sub_core_grids (ttnn.CoreRangeSet, optional): Subcore grids to use for the operation. Defaults to `None`, which will use all cores.{3}

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
                  - TILE{2}

            The output tensor will be in TILE layout and have the same dtype as the :attr:`input_tensor`.
            Exception: for sum and mean, 4D ROW_MAJOR BFLOAT16/FLOAT32 inputs with INTERLEAVED memory
            config reduced along the last (-1) or second-to-last (-2) dimension preserve ROW_MAJOR layout.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width, Height, and ND sharding
            - Output sharding will mirror the input
        )doc",
        op_name,
        qualified_name,
        int32_row,
        fast_approx_kwarg);
}

// Wrapper that detects explicit use of the deprecated 'correction' parameter and
// emits a Python DeprecationWarning before forwarding to the real implementation.
// The binding-layer type is std::optional<bool> (default nb::none()) so we can
// distinguish "user passed correction=True" from "used the default".
// This whole function can be removed when the deprecated 'correction' parameter is removed.
template <auto Func>
Tensor generic_reduction_with_deprecated_correction(
    const Tensor& input_tensor,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    std::optional<bool> correction,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (correction.has_value()) {
        // Re-acquire the GIL: this function is registered via bind_function<>(),
        // which applies nb::call_guard<nb::gil_scoped_release>(), so the GIL is
        // released by default. PyErr_WarnEx / PyExc_DeprecationWarning are Python
        // C API and require the GIL.
        nb::gil_scoped_acquire acquire;
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

// sum and mean expose an extra 'fast_and_approximate_mode' opt-in that min/max do not, so they need a
// wrapper with the trailing accurate flag instead of the shared generic_reduction_with_deprecated_correction<>.
// Same deprecated-correction handling; Func is &ttnn::sum or &ttnn::mean (identical signatures).
template <auto Func>
Tensor generic_reduction_fast_mode_with_deprecated_correction(
    const Tensor& input_tensor,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim,
    bool keepdim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    std::optional<bool> correction,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool fast_and_approximate_mode) {
    if (correction.has_value()) {
        nb::gil_scoped_acquire acquire;
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
        sub_core_grids,
        fast_and_approximate_mode);
}

inline void bind_generic_reductions(nb::module_& mod) {
    const auto sum_doc =
        get_generic_reduction_doc("sum", "ttnn.sum", /*int32_supported=*/true, /*has_fast_approximate_mode=*/true);
    ttnn::bind_function<"sum">(
        mod,
        sum_doc.c_str(),
        &generic_reduction_fast_mode_with_deprecated_correction<&ttnn::sum>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        // fast_and_approximate_mode=false (default) selects the accurate fp32 SFPU reduce; true selects the FPU. No
        // effect for non-fp32.
        nb::arg("fast_and_approximate_mode") = false);

    const auto mean_doc =
        get_generic_reduction_doc("mean", "ttnn.mean", /*int32_supported=*/false, /*has_fast_approximate_mode=*/true);
    ttnn::bind_function<"mean">(
        mod,
        mean_doc.c_str(),
        &generic_reduction_fast_mode_with_deprecated_correction<&ttnn::mean>,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scalar") = 1.0f,
        nb::arg("correction") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        // fast_and_approximate_mode=false (default) selects the accurate fp32 SFPU reduce; true selects the FPU. No
        // effect for non-fp32.
        nb::arg("fast_and_approximate_mode") = false);

    const auto max_doc = get_generic_reduction_doc("max", "ttnn.max", /*int32_supported=*/true);
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

    const auto min_doc = get_generic_reduction_doc("min", "ttnn.min", /*int32_supported=*/true);
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
