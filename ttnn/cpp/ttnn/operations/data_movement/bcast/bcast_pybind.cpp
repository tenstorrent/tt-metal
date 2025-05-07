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
        R"doc(bcast(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, math_op[ADD, SUB, MUL],  dim: Optional[int] = None, memory_config: Optional[MemoryConfig] = std::nullopt, output_tensor: Optional[Tensor]) -> ttnn.Tensor

            Perform a binary elementwise operation ``math_op`` between tensors ``input_a`` and ``input_b``, where values from tensor ``input_b`` are broadcast.

            Let tensor ``input_a`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``input_b`` shape ``[W1, Z1, Y1, X1]``. ``dim`` determines the type of broadcast performed.

            For ``dim=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

            For ``dim=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

            For ``dim=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

            Both input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
                "input_b", "Input tensor to broadcast", "Tensor", "Tensor of shape [W1, Z1, Y1, X1]", "Yes"
                "math_op", "Aggregating math operation", " BcastOpMath", "ADD, SUB, MUL", "Yes"
                "dim", "Dimension on which to broadcast", "BcastOpDim", "W, H, HW", "Yes"
                "memory_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional preallocated output tensor", "Tensor", "Default is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"

            Args:
                * :attr:`input_tensor_a`: First Input Tensor for bcast.
                * :attr:`input_tensor_b`: Second Input Tensor for bcast.
                * :attr:`math_op`: Operation to be performed during broadcasting.
                * :attr:`dim`: the dimension to reduce. If None, the bcast of the flattened input is returned

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`output_tensor`: Preallocated output tensor

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
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id, input_tensor_a, input_tensor_b, bcast_op, bcast_dim, memory_config, output_tensor);
            },
            py::arg("input_a").noconvert(),
            py::arg("input_b").noconvert(),
            py::arg("math_op"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::data_movement::detail
