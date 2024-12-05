// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "convert_to_chw.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::cnn::detail {

namespace py = pybind11;

void bind_convert_to_chw(py::module& module) {
    using OperationType = decltype(ttnn::experimental::convert_to_chw);

    const auto doc = R"doc(TODO!)doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::convert_to_chw,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               uint8_t queue_id) { return self(queue_id, input, memory_config, dtype); },
            py::arg("input"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::cnn::detail
