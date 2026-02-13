// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"

#include "expand.hpp"

namespace ttnn::operations::data_movement {

void bind_expand(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a new tensor where singleton dimensions are expanded to a larger side.
        Unlike :func:`torch.expand`, this function is not zero-cost and perform a memory copy to create the expanded tensor. This is due to `ttnn.Tensor`'s lack of strided tensor support.

        Args:
            * :attr:`input`: The tensor to be expanded.
            * :attr:`output_shape`: The desired output shape.
            * :attr:`memory_config`: The memory configuration for the expanded tensor.

        Requirements:
            like torch.expand:
                only size 1 dimensions can be expanded in the output shape
                -1 or the original shape size can be used to indicate that dimension should not have an expansion
                The output shape must have the same or higher dimensions than the input shape

        )doc";

    ttnn::bind_function<"expand">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::expand,
            nb::arg("input_tensor"),
            nb::arg("output_shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
