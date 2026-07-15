// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

#include "roll_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/roll/roll.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_roll(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Performs circular shifting of elements along the specified dimension(s).

        Args:
            input_tensor: A tensor whose elements will be rolled.
            shifts: The number of places by which elements are shifted. Can be an integer or a list of integers (one per dimension).
            dim: The dimension(s) along which to roll. If shifts is a list, then dim must be a list of the same length as shifts.
        )doc";

    ttnn::bind_function<"roll">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int>&,
                const ttsl::SmallVector<int>&,
                const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int, int, const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int, const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
