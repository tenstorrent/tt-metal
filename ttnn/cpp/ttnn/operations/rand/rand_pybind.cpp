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

        - DataType.float32 / bfloat16 / bfloat4_b / bfloat8_b:
            Floating-point values in range [0.0, 1.0)

        - Integer data types:
            Not supported for uniform random generation.

        Args:
            shape (list[int]) - a list of integers defining the shape of the output tensor.

        Keyword args:
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to ttnn.bfloat16.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to ttnn.TILE_LAYOUT.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            low (float, optional): The lower bound of the range (inclusive).
            high (float, optional): The upper bound of the range (exclusive).
            seed (int, optional): An optional seed to initialize the random number generator
                                for reproducible results. Defaults to 0.

        Returns:
            ttnn.Tensor: A tensor with specified shape, dtype, and layout containing random values.

        Example:
            >>> input_tensor_a = ttnn.rand([N,N], dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG )
            >>> input_tensor_b = ttnn.rand((N, N), device=device)
        )doc";

    using OperationType = decltype(ttnn::rand);
    bind_registered_operation(
        pymodule,
        ttnn::rand,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Shape& shape,
               MeshDevice& device,
               const DataType dtype,
               const Layout layout,
               const MemoryConfig& memory_config,
               float from,
               float to,
               uint32_t seed,
               QueueId queue_id) {
                return self(queue_id, shape, device, dtype, layout, memory_config, from, to, seed);
            },
            py::arg("shape"),
            py::arg("device"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("layout") = Layout::TILE,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
            py::arg("low") = 0.0f,
            py::arg("high") = 1.0f,
            py::arg("seed") = 0,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::rand
