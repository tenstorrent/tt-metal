// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::copy {

namespace {

void bind_global_typecast(nb::module_& mod) {
    const char* doc = R"doc(
Applies typecast to :attr:`input_tensor`.

Args:
    * :attr:`input_tensor` (ttnn.Tensor): input tensors must be on device, in ROW MAJOR or TILE layout
    * :attr:`dtype` (ttnn.DataType): output data type, must be one of: BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT32, INT32, UINT16

Keyword Args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
    * :attr:`output_tensor` (Optional[ttnn.Tensor]): Preallocated tensor to store the output.
    * :attr:`sub_core_grids` (Optional[CoreRangeSet]): Sub-core grids for the operation.

Returns:
    ttnn.Tensor: The tensor with the updated data type. Output tensor will be on device, in same layout, and have the given data type.

Example::

    >>> tensor = ttnn.typecast(input_tensor, ttnn.uint16)
)doc";

    mod.def(
        "typecast",
        &ttnn::typecast,
        doc,
        nb::arg("input_tensor"),
        nb::arg("dtype"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace

void py_module(nb::module_& mod) { bind_global_typecast(mod); }

}  // namespace ttnn::operations::copy
