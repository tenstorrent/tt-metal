// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/plusone/plusone.hpp"

namespace ttnn::operations::experimental::plusone::detail {
void bind_experimental_plusone_operation(nb::module_& mod) {
    auto doc =
        R"doc(plus_one(input_tensor: ttnn.Tensor) -> ttnn.Tensor

            Returns input tensor elements increased by 1.
            Input tensor must have UINT32 data type, ROW_MAJOR layout, and 1-D shape.
            This op only gives decent performance for small tensors (up to 100 elements).
            Specify the core to use in the sub_core_grids argument.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.add(input_tensor, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor for plusone.

        )doc";

    using OperationType = decltype(ttnn::plus_one);
    bind_registered_operation(
        mod,
        ttnn::plus_one,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) { return self(input_tensor, sub_core_grids); },
            nb::arg("input_tensor").noconvert(),
            nb::arg("sub_core_grids") = nb::none()});
}

}  // namespace ttnn::operations::experimental::plusone::detail
