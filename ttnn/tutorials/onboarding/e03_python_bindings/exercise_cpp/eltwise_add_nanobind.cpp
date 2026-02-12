// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Bind the operation to Python

#include <nanobind/nanobind.h>
#include "eltwise_add.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

NB_MODULE(_e03_exercise, mod) {
    mod.doc() = "E03 Exercise: Element-wise addition";

    ttnn::bind_registered_operation(
        mod,
        ttnn::e03_eltwise_add,
        R"doc(e03_eltwise_add(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor

        Element-wise addition of two tensors.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::e03_eltwise_add)& self, const ttnn::Tensor& a, const ttnn::Tensor& b)
                -> ttnn::Tensor { return self(a, b); },
            nb::arg("a"),
            nb::arg("b")});
}
