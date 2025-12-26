// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "tilize_untilize.hpp"
#include "device/kernels/op_types.hpp"

namespace ttnn::operations::reduction::tilize_untilize {

namespace nb = nanobind;

void py_bind_tilize_untilize(nb::module_& module) {
    const auto doc = R"doc(Tilize-Untilize operation that serves as a template for compute operations.

This operation converts row-major input to tiled format, performs computation,
and converts back to row-major output.

Args:
    input_tensor: Input tensor in ROW_MAJOR layout (4D, NCHW format)
    output_memory_config: Memory configuration for output (default: DRAM)
    output_dtype: Output data type (default: same as input)
    memory_config: Memory configuration (default: same as input)
    op_type: Operation type (default: IDENTITY). Options:
        - IDENTITY: Pass-through (tilize -> untilize)
        - REDUCE_W_SUM: Sum reduction along width
        - REDUCE_W_MAX: Max reduction along width
        - REDUCE_W_AVG: Average reduction along width
    scaler: Scaler value for reduction operations (default: 1.0)

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
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("output_memory_config") = nb::none(),
            nb::arg("output_dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("op_type") = ttnn::operations::reduction::OpType::IDENTITY,
            nb::arg("scaler") = 1.0f});
}

}  // namespace ttnn::operations::reduction::tilize_untilize
