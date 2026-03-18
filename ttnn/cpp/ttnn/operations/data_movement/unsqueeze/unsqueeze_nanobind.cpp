// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_unsqueeze(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a tensor unsqueezed at the specified dimension

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.rand((1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.unsqueeze(input_tensor, 2) # tensor of shape (1,1,1,256), where at dimension 2 we added a new dim of size 1

        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to unsqueeze (add a new dimension of size 1)
        )doc";

    ttnn::bind_function<"unsqueeze">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int>(&ttnn::unsqueeze), nb::arg("input_tensor"), nb::arg("dim")));
}

}  // namespace ttnn::operations::data_movement
