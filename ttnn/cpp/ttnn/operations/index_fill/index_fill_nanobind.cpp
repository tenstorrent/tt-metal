// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "index_fill.hpp"

namespace nb = nanobind;

namespace ttnn::operations::index_fill {

void bind_index_fill_operation(nb::module_& mod) {
    const char* doc =
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

    mod.def(
        "index_fill",
        &ttnn::index_fill,
        doc,
        nb::arg("input"),
        nb::arg("dim"),
        nb::arg("index"),
        nb::arg("value"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::index_fill
