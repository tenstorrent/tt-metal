// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/device/reshape_op.hpp"
#include "ttnn/experimental/jit/context.hpp"
#include "ttnn/types.hpp"
#pragma optimize("", off)
namespace ttnn::operations::data_movement {

void py_bind_reshape(pybind11::module& module) {
    module.def(
        "experimental_reshape",
        [](const ttnn::Tensor& input_tensor,
           int W,
           int Z,
           int Y,
           int X,
           const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
            // Create shapes from the individual parameters
            ttnn::Shape logical_shape{W, Z, Y, X};
            ttnn::Shape padded_shape{W, Z, Y, X};
            auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

            // Create ReshapeDeviceOperation struct
            ReshapeDeviceOperation reshape_op{logical_shape, padded_shape, output_mem_config};

            // Use Context to add a node with the args
            auto& context = ttnn::experimental::jit::Context::instance();

            // Create inputs vector for the node
            std::vector<ttnn::Tensor> inputs = {input_tensor};

            // Create shared_ptr to hold the args
            auto args_ptr = std::make_shared<ReshapeDeviceOperation>(reshape_op);

            // Add node to context
            auto node_id = context.create_node(
                inputs,
                "ttnn::reshape_on_device",
                std::static_pointer_cast<ttnn::experimental::jit::IDeviceOperation>(args_ptr));

            // For lazy JIT: return a tensor with computed specs that can be used as input to other operations
            auto output_specs = args_ptr->compute_output_specs(inputs);

            // Create output tensor with the computed specs and same storage/topology
            auto output_tensor = Tensor(input_tensor.storage(), output_specs[0], input_tensor.tensor_topology());
            output_tensor = tt::tt_metal::set_tensor_id(output_tensor);
            output_tensor.set_producer_node(node_id);

            return output_tensor;
        },
        py::arg("input_tensor"),
        py::arg("W"),
        py::arg("Z"),
        py::arg("Y"),
        py::arg("X"),
        py::kw_only(),
        py::arg("memory_config") = std::nullopt,
        R"doc(reshape_on_device(input_tensor: ttnn.Tensor, W: int, Z: int, Y: int, X: int, *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

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
            >>> output = ttnn.reshape_on_device(tensor, 1, 1, 2, 2)

        )doc");
}

}  // namespace ttnn::operations::data_movement
