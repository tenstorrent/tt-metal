// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_reshape(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int W,
               int Z,
               int Y,
               int X,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, W, Z, Y, X, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("W"),
            py::arg("Z"),
            py::arg("Y"),
            py::arg("X"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
            }
        );
}

}  // namespace detail


void py_bind_reshape(pybind11::module& module) {
    detail::bind_reshape(
        module,
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
            * :attr:`queue_id` (Optional[uint8]): command queue id

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.reshape(tensor, 1, 1, 2, 2)

        )doc");
}

}  // namespace ttnn::operations::data_movement
