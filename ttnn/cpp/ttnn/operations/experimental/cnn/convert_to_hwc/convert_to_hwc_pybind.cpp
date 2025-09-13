// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "convert_to_hwc.hpp"
#include "cpp/ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::cnn::detail {

namespace py = pybind11;

void bind_convert_to_hwc(py::module& module) {
    using OperationType = decltype(ttnn::experimental::convert_to_hwc);

    const auto doc = R"doc(
    Convert a tensor from CHW channel ordering to HWC channel ordering.

    The input tensor is expected to be in row-major layout and width-sharded in L1 or DRAM. The output is a row-major height-sharded tensor.
    )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::convert_to_hwc,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"), py::kw_only(), py::arg("memory_config") = std::nullopt, py::arg("dtype") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::cnn::detail
