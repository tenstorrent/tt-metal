// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full.hpp"
#include "pybind11/cast.h"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"

namespace ttnn::operations::full {
void bind_full_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the specified shape and fills it with the specified scalar value.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            fill_value (float or int): The value to fill the tensor with.
            any (ttnn.tensor): Any input tensor with desired device and data types for output tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: A filled tensor of specified shape and value.

        Example:
            >>> any = ttnn.zeros(shape=(2, 2), dtype=ttnn.bfloat16)
            >>> filled_tensor = ttnn.moreh_full([2, 2], any, 7.0, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[7.0,  7.0],
                            [7.0,  7.0]]]], shape=Shape([2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        ttnn::moreh_full.base_name());

    using FullType = decltype(ttnn::moreh_full);
    bind_registered_operation(
        module,
        ttnn::moreh_full,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("any"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::full
