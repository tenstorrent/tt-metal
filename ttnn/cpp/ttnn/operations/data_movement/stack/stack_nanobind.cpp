// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stack_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/stack/stack.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_stack(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Stacks tensors along a new dimension.

        Args:
            * :attr:`input_tensors`: List of tensors to stack.
            * :attr:`dim`: Dimension along which to stack.

        Example:
           >>> input_tensor = ttnn.from_torch(torch.randn((2, 2), dtype=torch.bfloat16), device=device)
           >>> output = ttnn.stack((input_tensor,input_tensor), 1)

        )doc";

    ttnn::bind_function<"stack">(mod, doc, ttnn::overload_t(&ttnn::stack, nb::arg("input_tensors"), nb::arg("dim")));
}

}  // namespace ttnn::operations::data_movement
