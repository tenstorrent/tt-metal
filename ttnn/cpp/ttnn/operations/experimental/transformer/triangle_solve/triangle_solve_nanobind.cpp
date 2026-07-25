// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "triangle_solve_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

#include "triangle_solve.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_triangle_solve(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Per-tile forward-substitution triangle solve of  L X = RHS  for a single 32x32 tile.

        L is a NEGATED unit-lower-triangular matrix: the caller pre-negates the strict-lower
        entries (the diagonal is an implicit 1 and the upper triangle is ignored). The solve is
        done on the SFPU.

        Args:
            l_neg (ttnn.Tensor): [1, 1, 32, 32] TILE bf16 — negated unit-lower-triangular L
            rhs (ttnn.Tensor):   [1, 1, 32, 32] TILE bf16 — right-hand-side matrix

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).

        Returns:
            ttnn.Tensor: X [1, 1, 32, 32] bf16 — the solution of  L X = RHS
        )doc";

    ttnn::bind_function<"triangle_solve", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::triangle_solve,
        nb::arg("l_neg"),
        nb::arg("rhs"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
