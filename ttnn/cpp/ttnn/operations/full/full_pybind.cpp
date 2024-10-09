// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <variant>

#include "full.hpp"
#include "pybind11/cast.h"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::full {

void bind_full_operation(py::module& module) {
    using FullType = decltype(ttnn::full_2);
    bind_registered_operation(
        module,
        ttnn::full_2,
        "Full Operation",
        ttnn::pybind_overload_t{
            [](const FullType& self,
               const std::vector<uint32_t>& shape,
               const std::variant<float, int> fill_value,
               const Tensor any,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(shape, fill_value, any, dtype, layout, memory_config);
            },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("any"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::full
