// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full_like.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"

namespace py = pybind11;

namespace ttnn::operations::full_like {

void bind_full_like_operation(py::module& module) {
    auto doc =
        R"doc(full_like(tensor: Tensor, fill_value: float or value, dtype: DataType, layout: Layout, memory_config: MemoryConfig) -> Tensor

    Create a tensor with the same shape of the given tensor and filled with given fill_value, with the specified `memory_config` and converting its data type to `dtype`.
    This operation only supports TILE_LAYOUT for now.

    Args:
        * :attr:`input`: The tensor has shape which will be based on to make the output tensor
        * :attr:`fill_value`: The value which will be used to fill the output tensor
        * :attr:`dtype`: The target data type of the output tensor.
        * :attr:`layout`: The target layout of the output tensor.
        * :attr:`memory_config`: The memory configuration for the output tensor.
    )doc";

    bind_registered_operation(
        module,
        ttnn::moreh_full_like,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("fill_value"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::full_like
