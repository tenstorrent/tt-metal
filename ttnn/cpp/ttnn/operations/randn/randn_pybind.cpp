// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "randn_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "randn.hpp"

namespace ttnn::operations::randn {
void bind_randn_operation(py::module& pymodule) {
    std::string doc =
        R"doc(
        Generates a tensor with the given shape, filled with random values from a standard normal distribution.
        Internally, this operation uses the Box-Muller transform to generate normally distributed random values.
        based on the specified data type:

        - DataType.float32 / bfloat16

        - Integer data types:
            Not supported for standard normal random generation.

        Args:
            shape (list[int]) - a list of integers defining the shape of the output tensor.

        Keyword args:
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to ttnn.bfloat16.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to ttnn.TILE_LAYOUT.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            seed (int, optional): An optional seed to initialize the random number generator for reproducible results. Defaults to 0.

        Returns:
            ttnn.Tensor: A tensor with specified shape, dtype, and layout containing random values.
        )doc";

    using OperationType = decltype(ttnn::randn);
    bind_registered_operation(
        pymodule,
        ttnn::randn,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Shape& shape,
               MeshDevice& device,
               const DataType dtype,
               const Layout layout,
               const MemoryConfig& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               uint32_t seed) {
                return self(shape, device, dtype, layout, memory_config, compute_kernel_config, seed);
            },
            py::arg("shape"),
            py::arg("device"),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("layout") = Layout::TILE,
            py::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("seed") = 0});
}
}  // namespace ttnn::operations::randn
