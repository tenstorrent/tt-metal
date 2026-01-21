// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "full.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"

namespace ttnn::operations::full {
void bind_full_operation(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the specified shape and fills it with the specified scalar value.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            fill_value (float or int): The value to fill the tensor with.
            device (ttnn.MeshDevice): The device on which the tensor will be allocated.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `ttnn.TILE_LAYOUT`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: A filled tensor of specified shape and value.

        Example:
            >>> device = ttnn.open_device(device_id=0)
            >>> filled_tensor = ttnn.moreh_full([2, 2], 7.0, device, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[7.0,  7.0],
                            [7.0,  7.0]]]], shape=Shape([2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        ttnn::moreh_full.base_name());

    bind_registered_operation(
        mod,
        ttnn::moreh_full,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("shape"),
            nb::arg("fill_value"),
            nb::arg("device"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("layout") = ttnn::TILE_LAYOUT,
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

}  // namespace ttnn::operations::full
