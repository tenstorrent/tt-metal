// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/core.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace core {

void py_module(py::module& module) {
    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const ttnn::Shape& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 1>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 2>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 3>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 4>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 5>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def("unsqueeze_to_4D", &ttnn::unsqueeze_to_4D, py::arg("tensor"));

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, Device*, const std::optional<MemoryConfig>&>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt);

    module.def(
        "to_device",
        py::overload_cast<const ttnn::Tensor&, DeviceMesh*, const std::optional<MemoryConfig>&>(
            &ttnn::operations::core::to_device),
        py::arg("tensor"),
        py::arg("device"),
        py::arg("memory_config") = std::nullopt);

    module.def("from_device", &ttnn::operations::core::from_device, py::arg("tensor"), py::arg("blocking") = true);

    module.def("deallocate", &ttnn::operations::core::deallocate, py::arg("tensor"), py::arg("force") = true);

    module.def(
        "to_memory_config",
        &ttnn::operations::core::to_memory_config,
        py::arg("tensor"),
        py::arg("memory_config"),
        py::arg("dtype") = std::nullopt);

    module.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return reallocate(input_tensor, memory_config); },
        py::arg("tensor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
Deallocates device tensor and returns a reallocated tensor

Args:
    * :attr:`input_tensor`: Input Tensor
    )doc");

    bind_registered_operation(
        module,
        ttnn::to_layout,
        R"doc(to_layout(tensor: ttnn.Tensor, layout: Layout, dtype: Optional[DataType] = None, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

    Organizes the `ttnn.Tensor` :attr:`tensor` into either ROW_MAJOR_LAYOUT or TILE_LAYOUT.  When requesting ROW_MAJOR_LAYOUT
    the tensor will be returned unpadded in the last two dimensions.   When requesting TILE_LAYOUT the tensor will be automatically
    padded where the width and height become multiples of 32.
    In the case where the layout is the same, the operation simply pad or unpad the last two dimensions depending on layout requested.

    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`layout`: the layout of either ttnn.ROW_MAJOR_LAYOUT or ttnn.TILE_LAYOUT.
        * :attr:`dtype`: the optional output data type.
        * :attr:`memory_config`: the optional output memory configuration.

    Example::
        >>> device_id = 0
        >>> device = ttnn.open_device(device_id=device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
        >>> print(tensor[0,0,:3])
        Tensor([ 1.42188, -1.25, -0.398438], dtype=bfloat16 ))doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::arg("layout"),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
