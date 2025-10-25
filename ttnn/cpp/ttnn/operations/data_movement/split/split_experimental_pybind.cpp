// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/split/device/split_op.hpp"
#include "ttnn/types.hpp"

#include "split.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_experimental_split(pybind11::module& module) {
    // Create a submodule for experimental functions
    auto split_submodule = module.def_submodule("jit_split", "Experimental split operation");

    // Add attributes to the submodule
    split_submodule.attr("version") = "LAZY_JIT";
    split_submodule.attr("python_fully_qualified_name") = "ttnn.jit_split";

    // Define the function in the submodule and store the function object
    auto split_func = split_submodule.def(
        "operation_function",
        [](const ttnn::Tensor& input_tensor,
           int num_splits,
           int dim,
           const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
            auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

            // Create SplitDeviceOperation struct
            SplitDeviceOperation split_op{num_splits, dim, output_mem_config};
            // Create inputs vector for the node
            std::vector<ttnn::Tensor> inputs = {input_tensor};
            // return ttnn::python_binding::bind_operation(inputs, "ttnn:jit_split",
            // std::make_shared<SplitDeviceOperation>(split_op));
            return inputs;
        },
        py::arg("input_tensor"),
        py::arg("num_splits"),
        py::arg("dim"),
        py::kw_only(),
        py::arg("memory_config") = std::nullopt,
        R"doc(split(input_tensor: ttnn.Tensor, num_splits: int, dim: int, *, Optional[ttnn.MemoryConfig] = None) -> List[ttnn.Tensor]

        Returns a list of split tensors from splitting the input tensor into num_splits parts along the specified dimension.
        This is a JIT version that builds a computation graph without immediate execution.

        Equivalent pytorch code:

        .. code-block:: python
            input_tensor = torch.rand(1, 1, 4, 8)
            split_tensors = torch.split(input_tensor, 2, dim=3)
            # Returns list of split tensors

        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`num_splits`: Number of splits to create.
            * :attr:`dim`: Dimension along which to split.

        Keyword Args:
            * :attr:`memory_config`: Memory Config of the output tensor

        Example:

            >>> tensor = ttnn.from_torch(torch.rand(1, 1, 4, 8), dtype=ttnn.bfloat16, device=device)
            >>> split_tensors = ttnn.experimental.split(tensor, 2, 3)
            >>> print(f"Number of splits: {len(split_tensors)}")

        )doc");

    split_submodule.attr("function") = split_func.attr("operation_function");
}

}  // namespace ttnn::operations::data_movement::detail
