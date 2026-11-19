// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
        Stacks tensors along a new dimension. All inputs must share shape, dtype and layout; the output
        has one more dimension than each input, with the number of inputs at position `dim`. The output
        currently lands in DRAM interleaved (concat's default), whatever the inputs' placement.

        Args:
            * :attr:`input_tensors`: List of tensors to stack.
            * :attr:`dim`: Position of the new dimension in the output, in [-(rank + 1), rank]; -1 and rank both mean a new last dimension.

        Equivalent pytorch code:

        .. code-block:: python

            output = torch.stack(input_tensors, dim)

        Example:
           >>> input_tensor = ttnn.from_torch(torch.randn((2, 2), dtype=torch.bfloat16), device=device)
           >>> output = ttnn.stack((input_tensor,input_tensor), 1)

        )doc";

    ttnn::bind_function<"stack">(mod, doc, &ttnn::stack, nb::arg("input_tensors"), nb::arg("dim"));
}

}  // namespace ttnn::operations::data_movement
