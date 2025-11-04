// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_pybind.hpp"

#include "clone.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::data_movement::clone {
void bind_clone_operation(py::module& module) {
    auto doc = R"doc(
        Clones the input tensor, creating a copy with the specified memory configuration and converting its data type to dtype. This operation does not alter the tensor's layout.

        Args:
            input (ttnn.Tensor): the input tensor to be cloned.

        Keyword Args:
            dtype (ttnn.DataType, optional): the target data type of the cloned tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the clone, options include DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): the configuration for the compute kernel. Defaults to `None`.

        Note:
            * ROW_MAJOR_LAYOUT: Returns the tensor unpadded in the last two dimensions.
            * TILE_LAYOUT: Pads the tensor to ensure its width and height are multiples of 32.
            * If the input's current layout matches the specified layout, padding adjustments are applied to the last two dimensions as necessary.

        Returns:


            ttnn.Tensor: the output tensor.



        Example:
            >>> tensor = ttnn.from_torch(torch.rand([1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.clone(tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    )doc";

    bind_registered_operation(
        module,
        ttnn::clone,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::data_movement::clone
