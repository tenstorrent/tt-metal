// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_reshape(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int W,
               int Z,
               int Y,
               int X,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(input_tensor, ttnn::SmallVector<int32_t>{W, Z, Y, X}, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("W"),
            nb::arg("Z"),
            nb::arg("Y"),
            nb::arg("X"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace detail

void bind_reshape(nb::module_& mod) {
    detail::bind_reshape(
        mod,
        ttnn::reshape_on_device,
        R"doc(reshape(input_tensor: ttnn.Tensor, W: int, Z: int, Y: int, X: int, *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Returns a tensor with the new shape of ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

        Equivalent pytorch code:

        .. code-block:: python
            input_tensor = torch.arange(4.)
            W = 1
            Z = 1
            Y = 2
            X = 2
            output_tensor = torch.reshape(input_tensor, (W, Z, Y, X))


        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`W`: W dim of tensor.
            * :attr:`Z`: Z dim of tensor.
            * :attr:`Y`: Y dim of tensor.
            * :attr:`X`: X dim of tensor.

        Keyword Args:
            * :attr:`memory_config`: Memory Config of the output tensor

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.reshape(tensor, 1, 1, 2, 2)

        )doc");
}

}  // namespace ttnn::operations::data_movement
