// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/unsqueeze/unsqueeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_unsqueeze(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a tensor unsqueezed at the specified dimension (Quasar / Metal 2.0, via the quasar reshape).

        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to unsqueeze (add a new dimension of size 1)
        )doc";

    ttnn::bind_function<"unsqueeze", "ttnn.experimental.quasar.">(
        mod,
        doc,
        nb::overload_cast<const ttnn::Tensor&, int>(&ttnn::operations::experimental::quasar::unsqueeze),
        nb::arg("input_tensor"),
        nb::arg("dim"));
}

}  // namespace ttnn::operations::experimental::quasar::detail
