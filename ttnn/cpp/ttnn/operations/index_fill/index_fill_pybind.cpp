// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index_fill.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::index_fill {

void bind_index_fill_operation(py::module& module) {
    auto doc =
        R"doc(index_fill(input: Tensor, dim: uint32, index: Tensor, value: int or float, memory_config: MemoryConfig) -> Tensor
    Create or fill a tensor with the given value, with the specified `memory_config`.
    This operation only supports ROW_MAJOR_LAYOUT for now.
    Args:
        * :attr:`input`: The tensor that we will operate on
        * :attr:`dim`: The dimension that we need to fill the value along.
        * :attr:`index`: The index that we need to fill the value in.
        * :attr:`value`: The value which will be used to fill the output tensor
        * :attr:`memory_config`: The memory configuration for the output tensor.
    )doc";

    bind_registered_operation(
        module,
        ttnn::index_fill,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("dim"),
            py::arg("index"),
            py::arg("value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::index_fill
