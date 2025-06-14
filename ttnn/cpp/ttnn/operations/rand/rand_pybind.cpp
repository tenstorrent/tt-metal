// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "rand.hpp"

namespace ttnn::operations::rand {
void bind_rand_operation(py::module& pymodule) {
    std::string doc =
        R"doc(
        Generates a tensor with the given shape, filled with random values from a uniform distribution.
        based on the specified data type:

        - DataType.uint8 / uint16 / uint32:
            Integer values: 0 or 1

        - DataType.float32 / bfloat16:
            Floating-point values in range [0.0, 1.0)

        - DataType.int32:
            Integer values: -1, 0, 1

        - DataType.bfloat4_b / bfloat8_b:
            Not supported for uniform random generation.

        Args:
            size (list[int]) - a list of integers defining the shape of the output tensor.

        Keyword args:
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to ttnn.bfloat16.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to ttnn.TILE_LAYOUT.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: A tensor with specified shape, dtype, and layout containing random values.

        Example:
            >>> input_tensor_a = ttnn.rand([N,N], dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG )
            >>> input_tensor_b = ttnn.rand((N, N), device=device)
        )doc";

    bind_registered_operation(
        pymodule,
        ttnn::rand,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("size"),
            py::kw_only(),
            py::arg("device") = std::nullopt,
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("layout") = Layout::TILE,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
        });
}
}  // namespace ttnn::operations::rand
