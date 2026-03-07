// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "randn_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"
#include "randn.hpp"

namespace ttnn::operations::randn {
void bind_randn_operation(nb::module_& mod) {
    std::string doc =
        R"doc(
        Generates a tensor with the given shape, filled with random values from a standard normal distribution.
        Internally, this operation uses the Box-Muller transform to generate normally distributed random values.

        Args:
            shape (list[int]): a list of integers defining the shape of the output tensor.

        Keyword Args:
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `ttnn.TILE_LAYOUT`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            seed (int, optional): An optional seed to initialize the random number generator for reproducible results. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            The output tensor supports the following data types and layouts:

            .. list-table:: Output Tensor
                :header-rows: 1

                * - dtype
                - layout
                * - FLOAT32
                - ROW_MAJOR, TILE
                * - BFLOAT16
                - ROW_MAJOR, TILE

        Memory Support:
            - Interleaved: DRAM and L1
            - Height, Width, Block, and ND Sharded: DRAM and L1
        )doc";

    using OperationType = decltype(ttnn::randn);
    bind_registered_operation(
        mod,
        ttnn::randn,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Shape& shape,
               MeshDevice& device,
               const DataType dtype,
               const Layout layout,
               const MemoryConfig& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               std::optional<uint32_t> seed) {
                return self(shape, device, dtype, layout, memory_config, compute_kernel_config, seed);
            },
            nb::arg("shape"),
            nb::arg("device"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("layout") = Layout::TILE,
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("seed") = std::nullopt});
}
}  // namespace ttnn::operations::randn
