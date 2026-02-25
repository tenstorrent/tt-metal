// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "sign.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

NB_MODULE(_e05_solution, mod) {
    mod.doc() = "E05 Solution: Sign operation (debugging exercise)";

    ttnn::bind_registered_operation(
        mod,
        ttnn::s05_sign,
        R"doc(s05_sign(input) -> Tensor

        Computes element-wise sign of the input tensor.
        output = sign(input) = -1 if input < 0, 0 if input == 0, 1 if input > 0
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::s05_sign)& self, const ttnn::Tensor& input) -> ttnn::Tensor { return self(input); },
            nb::arg("input")});
}
