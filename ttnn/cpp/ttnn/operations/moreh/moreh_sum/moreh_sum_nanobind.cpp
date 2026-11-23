// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_sum {
namespace {
Tensor moreh_sum_with_warning(
    const Tensor& input,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    {
        nb::gil_scoped_acquire gil;
        if (PyErr_WarnEx(
                PyExc_FutureWarning,
                "ttnn.moreh_sum is deprecated and will be removed after 2026-12-19. Use ttnn.sum instead.",
                2) < 0) {
            throw nb::python_error();
        }
    }
    return ttnn::moreh_sum(input, dim, keepdim, output, memory_config, compute_kernel_config);
}
}  // namespace

void bind_moreh_sum_operation(nb::module_& mod) {
    const auto* doc =
        "Deprecated in favor of ttnn.sum. Will be removed after 2026-12-19. With keepdim=False, moreh_sum keeps "
        "reduced H/W dims as size 1; ttnn.sum removes them.";

    ttnn::bind_function<"moreh_sum">(
        mod,
        doc,
        &moreh_sum_with_warning,
        nb::arg("input"),
        nb::arg("dim") = nb::none(),
        nb::kw_only(),
        nb::arg("keepdim") = false,
        nb::arg("output") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::moreh::moreh_sum
