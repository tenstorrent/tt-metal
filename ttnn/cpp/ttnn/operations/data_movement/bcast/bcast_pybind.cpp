// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void py_bind_bcast(py::module& module) {
    auto doc =
        R"doc(
        Performs a binary elementwise operation between tensors with broadcasting.

        Performs a binary elementwise operation ``math_op`` between tensors ``input_a`` and ``input_b``,
        where values from tensor ``input_b`` are broadcast according to the specified dimension.

        Let tensor ``input_a`` have shape [W0, Z0, Y0, X0] and tensor ``input_b`` shape [W1, Z1, Y1, X1].
        The ``dim`` parameter determines the type of broadcast performed:

        - For ``dim=BcastOpDim::W``: broadcast on dimension X. Y0 and Y1 must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).
        - For ``dim=BcastOpDim::H``: broadcast on dimension Y. X0 and X1 must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).
        - For ``dim=BcastOpDim::HW``: broadcast on dimensions X and Y. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold.

        Args:
            input_a (ttnn.Tensor): First input tensor with shape [W0, Z0, Y0, X0]. Must have BFLOAT16 data type.
            input_b (ttnn.Tensor): Second input tensor to broadcast with shape [W1, Z1, Y1, X1]. Must have BFLOAT16 data type.
            math_op (ttnn.BcastOpMath): Math operation to perform (ADD, SUB, or MUL).
            dim (ttnn.BcastOpDim): Dimension on which to broadcast (W, H, or HW).

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to interleaved in DRAM.
            output_tensor (Optional[ttnn.Tensor]): Preallocated output tensor. Defaults to None.

        Returns:
            ttnn.Tensor: Output tensor with BFLOAT16 data type.

        Example:

            >>> tensor_a = ttnn.from_torch(torch.ones(1, 1, 32, 64), dtype=ttnn.bfloat16, device=device)
            >>> tensor_b = ttnn.from_torch(torch.ones(1, 1, 32, 1), dtype=ttnn.bfloat16, device=device)
            >>> output = ttnn.bcast(tensor_a, tensor_b, ttnn.BcastOpMath.ADD, ttnn.BcastOpDim.W)
        )doc";

    using OperationType = decltype(ttnn::bcast);
    bind_registered_operation(
        module,
        ttnn::bcast,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               ttnn::BcastOpMath bcast_op,
               ttnn::BcastOpDim bcast_dim,
               std::optional<Tensor> output_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, bcast_op, bcast_dim, memory_config, output_tensor);
            },
            py::arg("input_a").noconvert(),
            py::arg("input_b").noconvert(),
            py::arg("math_op"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
