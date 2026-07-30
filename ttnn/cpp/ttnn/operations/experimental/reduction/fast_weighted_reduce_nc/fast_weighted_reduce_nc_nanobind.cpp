// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fast_weighted_reduce_nc_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/fast_weighted_reduce_nc.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_fast_weighted_reduce_nc(nb::module_& mod) {
    ttnn::bind_function<"fast_weighted_reduce_nc", "ttnn.experimental.">(
        mod,
        R"doc(
            Reduces `input` along `dim`, weighting each slice by `weight`, in a single pass:

                out[b][0][h][w] = sum_c input[b][c][h][w] * weight[b][c][h][0]

            `weight` carries one scalar per (batch, slice, row) and is broadcast along
            the last dim. It must be TILE layout with a logical last dim of 1; that is
            already the layout the hardware column broadcast reads, so no transpose or
            repeat is needed on the caller's side.

            Only `dim == 1` on a rank-4 bfloat16 input is implemented.

            Equivalent to `ttnn.sum(input * weight, dim=dim, keepdim=True)` without the
            full-size intermediate, and without reading `input` more than once.
        )doc",
        &ttnn::experimental::reduction::fast_weighted_reduce_nc,
        nb::arg("input").noconvert(),
        nb::arg("weight").noconvert(),
        nb::kw_only(),
        nb::arg("dim") = 1,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
