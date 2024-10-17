// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/experimental/pool/avgpool/avg_pool.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;
namespace ttnn::operations::experimental::pool {
namespace detail {

void bind_avg_pool2d(py::module& module) {
    auto doc = fmt::format("TODO");

    bind_registered_operation(
        module,
        ttnn::experimental::avg_pool2d,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("batch_size"),
            py::arg("input_h"),
            py::arg("input_w"),
            py::arg("channels"),
            py::arg("kernel_size"),
            py::arg("stride"),
            py::arg("padding"),
            py::arg("ceil_mode") = false,
            py::arg("count_include_pad") = true,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace detail
}  // namespace ttnn::operations::experimental::pool
