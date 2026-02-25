// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "sign.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

NB_MODULE(_e05_exercise, mod) {
    mod.doc() = "E05 Exercise: Sign operation (has a bug that causes hang!)";

    ttnn::bind_registered_operation(
        mod,
        ttnn::e05_sign,
        R"doc(e05_sign(input) -> Tensor

        BUGGY: This kernel will hang.
        Use tt-triage to diagnose the issue.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::e05_sign)& self, const ttnn::Tensor& input) -> ttnn::Tensor { return self(input); },
            nb::arg("input")});
}
