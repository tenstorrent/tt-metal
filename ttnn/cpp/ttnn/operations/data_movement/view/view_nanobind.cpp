// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "view_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_view(nb::module_& mod) {
    const auto* doc =
        R"doc(
        This is a 0 cost view operation that returns the same tensor that was passed to it but with a new shape

        Note: The following conditions must be met:
            * the memory is stored on the device
            * the last dimension must not change
            * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

        Args:
            * input_tensor: Input Tensor.
            * new_shape: New shape of tensor.

        Returns:
            ttnn.Tensor: a reference to the input tensor but with the new shape.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((2, 1, 1, 1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.view(tensor, (2, 1, 4))

        )doc";

    // Bind first overload: view(Tensor, Shape)
    ttnn::bind_function<"view">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttnn::Shape&>(&ttnn::view),
            nb::arg("input_tensor"),
            nb::arg("shape")));

    // Bind second overload: view(Tensor, SmallVector<int32_t>)
    ttnn::bind_function<"view">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, tt::stl::Span<const int32_t>>(&ttnn::view),
            nb::arg("input_tensor"),
            nb::arg("shape")));
}
}  // namespace ttnn::operations::data_movement
