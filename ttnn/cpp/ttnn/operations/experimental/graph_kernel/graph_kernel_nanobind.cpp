// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_kernel_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/graph_kernel/graph_kernel.hpp"

namespace ttnn::operations::experimental::graph_kernel_op::detail {

void bind_experimental_graph_kernel_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Experimental graph kernel.

            Takes an arbitrary number of input tensors and a string describing the graph to run
            over them. All inputs must be interleaved device tensors on the same device.

            Current basis behaviour: returns a new tensor with the same spec as ``inputs[0]``,
            containing a copy of ``inputs[0]``. Every input is bound into the device program so
            the kernel can be extended to consume them; ``text`` participates in the program
            cache hash.

            Args:
                * :attr:`inputs`: List of input tensors (at least one).
                * :attr:`text`: Graph description string.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc";

    ttnn::bind_function<"graph_kernel">(
        mod, doc, &ttnn::operations::experimental::graph_kernel, nb::arg("inputs"), nb::arg("text"));
}

}  // namespace ttnn::operations::experimental::graph_kernel_op::detail
