// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"

namespace ttnn::operations::moreh::moreh_mean {
namespace {
Tensor moreh_mean_with_warning(
    const Tensor& input,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<uint32_t>& divisor,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    {
        nb::gil_scoped_acquire gil;
        if (PyErr_WarnEx(
                PyExc_FutureWarning,
                "ttnn.moreh_mean is deprecated and will be removed after 2026-12-19. Use ttnn.mean instead.",
                2) < 0) {
            throw nb::python_error();
        }
    }
    return ttnn::moreh_mean(input, dim, keepdim, divisor, output, memory_config, compute_kernel_config);
}
}  // namespace

void bind_moreh_mean_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_mean">(
        mod,
        "Deprecated in favor of ttnn.mean. Will be removed after 2026-12-19. With keepdim=False, moreh_mean keeps "
        "reduced H/W dims as size 1; ttnn.mean removes them.",
        &moreh_mean_with_warning,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::arg("divisor") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::moreh::moreh_mean
