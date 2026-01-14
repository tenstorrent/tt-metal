// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "prefix_scan.hpp"

namespace ttnn::operations::experimental::ssm::detail {

void bind_prefix_scan(nb::module_& mod) {
    const auto* const doc =
        R"doc(Performs a prefix scan to produce the SSM hidden states across an entire sequence. All input and output tensors are expected to be shape [1, 1, L, 2EN]. Values of 2EN and L can be any multiple of 32.)doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::prefix_scan,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("a"),
            nb::arg("bx"),
            nb::arg("h_prev"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("math_fidelity") = nb::none()});
}

}  // namespace ttnn::operations::experimental::ssm::detail
