// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Bind the operation to Python

#include <nanobind/nanobind.h>
#include "matmul_add.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

NB_MODULE(_e04_exercise, mod) {
    mod.doc() = "E04 Exercise: Matmul + add";

    ttnn::bind_registered_operation(
        mod,
        ttnn::e04_matmul_add,
        R"doc(e04_matmul_add(a, b, c) -> Tensor

        Compute a @ b + c (matmul followed by element-wise add).
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::e04_matmul_add)& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& b,
               const ttnn::Tensor& c) -> ttnn::Tensor { return self(a, b, c); },
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("c")});
}
