// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full_like_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"
#include "full_like.hpp"

namespace py = pybind11;


namespace ttnn::operations::full_like {

void bind_full_like_operation(py::module& module) {
using FullLikeType = decltype(ttnn::full_like_2);
bind_registered_operation(
    module,
    ttnn::full_like_2,
    "Full like Operation",
    ttnn::pybind_overload_t{
        [](const FullLikeType& self,
            const ttnn::Tensor& tensor,
            const float fill_value,
            const std::optional<DataType>& dtype,
            const std::optional<Layout>& layout,
            const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
            return self(tensor, fill_value, dtype, layout, memory_config);
        },
        py::arg("tensor"),
        py::arg("fill_value"),
        py::arg("dtype") = std::nullopt,
        py::arg("layout") = std::nullopt,
        py::arg("memory_config") = std::nullopt},
    ttnn::pybind_overload_t{
        [](const FullLikeType& self,
            const ttnn::Tensor& tensor,
            const int fill_value,
            const std::optional<DataType>& dtype,
            const std::optional<Layout>& layout,
            const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
            return self(tensor, fill_value, dtype, layout, memory_config);
        },
        py::arg("tensor"),
        py::arg("fill_value"),
        py::arg("dtype") = std::nullopt,
        py::arg("layout") = std::nullopt,
        py::arg("memory_config") = std::nullopt});
}

}
