// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "view_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "view.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::reshape::detail {

void bind_view(nb::module_& mod) {
    const auto* doc = R"doc(
        Note:
        - It is recommended to use ttnn.reshape if you are not sure which operation to use
        - If this is the functionality required for your application, it will be called by ttnn.reshape
        - The following conditions must be met for the function not to corrupt your data:
            * the last dimension must not change
            * In Layout::TILE the second last two dimensions must not change OR there is no padding on the second last dimension
        Args:
            * input_tensor: Input Tensor.
            * shape: Shape of tensor.
        Returns:
            ttnn.Tensor: the output tensor with the new shape.
        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((2, 1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.experimental.view(tensor, (2, 1, 1, 4))
        )doc";

    bind_registered_operation(
        mod,
        ttnn::experimental::view,
        doc,
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::view)& self, ttnn::Tensor& input_tensor, int N, int C, int H, int W) {
                return self(input_tensor, ttnn::SmallVector<int>{N, C, H, W});
            },
            nb::arg("input_tensor"),
            nb::arg("N"),
            nb::arg("C"),
            nb::arg("H"),
            nb::arg("W"),
        },
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::view)& self,
               ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int32_t>& shape) { return self(input_tensor, shape); },
            nb::arg("input_tensor"),
            nb::arg("shape"),
        });
}

}  // namespace ttnn::operations::experimental::reshape::detail
