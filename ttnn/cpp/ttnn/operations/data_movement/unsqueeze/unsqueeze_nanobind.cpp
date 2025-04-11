// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_unsqueeze(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int dim) -> ttnn::Tensor {
                return self(input_tensor, dim);
            },
            nb::arg("input_tensor"),
            nb::arg("dim")});
}

}  // namespace detail

void bind_unsqueeze(nb::module_& mod) {
    detail::bind_unsqueeze(
        mod,
        ttnn::unsqueeze,
        R"doc(unsqueeze(input_tensor: ttnn.Tensor,  dim: int) -> ttnn.Tensor

        Returns a tensor unsqueezed at the specified dimension

        Equivalent pytorch code:

        .. code-block:: python
            input_tensor = torch.rand((1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.unsqueeze(input_tensor, 2) # tensor of shape (1,1,1,256), where at dimension 2 we added a new dim of size 1



        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to unsqueeze (add a new dimension of size 1)


        )doc");
}

}  // namespace ttnn::operations::data_movement
