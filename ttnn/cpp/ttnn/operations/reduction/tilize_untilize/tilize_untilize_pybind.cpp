// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "tilize_untilize.hpp"

namespace ttnn::operations::reduction::tilize_untilize {

namespace py = pybind11;

void py_bind_tilize_untilize(py::module& module) {
    const auto doc = R"doc(Tilize-Untilize operation that serves as a template for compute operations.

This operation converts row-major input to tiled format, performs computation
(identity in this template), and converts back to row-major output.

Args:
    input_tensor: Input tensor in ROW_MAJOR layout (4D, NCHW format)
    output_memory_config: Memory configuration for output (default: DRAM)
    output_dtype: Output data type (default: same as input)

Returns:
    Output tensor with same shape as input in ROW_MAJOR layout

Constraints:
    - Input must be 4D tensor
    - Input must be in ROW_MAJOR layout
    - Input must be interleaved (not sharded)
    - Input must be on device
    - Height and width must be multiples of 32
    - Supported dtypes: BFLOAT16, FLOAT32)doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::tilize_untilize,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("output_memory_config") = std::nullopt,
            py::arg("output_dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::tilize_untilize
