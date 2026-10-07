// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_practice_quasar_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "binary_practice_quasar.hpp"

namespace ttnn::operations::experimental::binary_practice::detail {
void bind_binary_practice_quasar_operation(nb::module_& mod) {
    // Default namespace "ttnn.", so Python sees it as ttnn.binary_practice_quasar.
    ttnn::bind_function<"binary_practice_quasar">(
        mod,
        R"doc(
        Practice binary op for Quasar (Metal 2.0 / DataflowBuffer): out = a + b.

        Both inputs must be 2D, BFLOAT16, TILE layout, interleaved, and the same shape (no broadcast).
        )doc",
        &ttnn::operations::experimental::binary_practice_quasar,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert());
}
}  // namespace ttnn::operations::experimental::binary_practice::detail
