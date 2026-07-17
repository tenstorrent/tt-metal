// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze_nanobind.hpp"

#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"

namespace ttnn::operations::data_movement {

namespace {
// nanobind variant/optional casters perform the Python -> C++ conversion at
// the wrapper boundary (GIL held), so the body runs with the GIL released
// (call_guard applied by bind_function).
ttnn::Tensor squeeze_wrapper(
    const ttnn::Tensor& input_tensor, std::optional<std::variant<int, ttsl::SmallVector<int>>> dim) {
    if (!dim.has_value()) {  // None
        return ttnn::squeeze(input_tensor);
    }
    return std::visit([&](const auto& d) { return ttnn::squeeze(input_tensor, d); }, *dim);
}
}  // namespace

void bind_squeeze(nb::module_& mod) {
    ttnn::bind_function<"squeeze">(
        mod,
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
        )doc",
        &squeeze_wrapper,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none()  // Default value is None
    );
}

}  // namespace ttnn::operations::data_movement
