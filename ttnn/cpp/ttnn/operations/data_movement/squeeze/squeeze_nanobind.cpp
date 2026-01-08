// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_squeeze(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const nb::object& dim)
                -> ttnn::Tensor {
                if (dim.is_none()) {  // None
                    return self(input_tensor);
                }
                if (nb::isinstance<nb::int_>(dim)) {  // int
                    return self(input_tensor, nb::cast<int>(dim));
                }
                if (nb::isinstance<nb::list>(dim)) {  // List[int]
                    auto dims = nb::cast<ttnn::SmallVector<int>>(dim);
                    return self(input_tensor, dims);
                }
                throw std::invalid_argument("dim must be an int, a list of ints, or None");
            },
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none()  // Default value is None
        });
}

}  // namespace detail

void bind_squeeze(nb::module_& mod) {
    detail::bind_squeeze(
        mod,
        ttnn::squeeze,
        R"doc(
        Returns a tensor with the specified dimensions squeezed. If `dim` is not provided, all dimensions of size 1 will be squeezed. If `dim` is an integer, only the specified dimension will be squeezed. If `dim` is a list of integers, all specified dimensions will be squeezed.

        If a specified dimension in `dim` does not have size 1, it will be ignored.

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.rand((1,1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.squeeze(input_tensor, 2) # tensor of shape (1,1,256), where at dimension 2 we removed it

        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to squeeze
        )doc");
}

}  // namespace ttnn::operations::data_movement
