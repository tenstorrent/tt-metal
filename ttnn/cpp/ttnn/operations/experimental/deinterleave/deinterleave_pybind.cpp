// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deinterleave_pybind.hpp"

#include "deinterleave.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::experimental::deinterleave {
void bind_deinterleave_operation(py::module& module) {
    auto doc = R"doc(deinterleave(input: Tensor, dtype: DataType, memory_config: MemoryConfig) -> Tensor

    Deinterleaves the input, creating a copy with the specified `memory_config` and converting its data type to `dtype`.
    This operation does not alter the tensor's layout.
    - ROW_MAJOR_LAYOUT: Returns the tensor unpadded in the last two dimensions.
    - TILE_LAYOUT: Pads the tensor to ensure its width and height are multiples of 32.
    If the input's current layout matches the specified layout, padding adjustments are applied to the last two dimensions as necessary.

    Args:
        * :attr:`input`: The tensor to be Deinterleaved.
        * :attr:`dtype`: The target data type of the Deinterleaved tensor.
        * :attr:`memory_config`: The memory configuration for the Deinterleave, options include DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG.
        * :attr:`compute_kernel_config`: The configuration for the compute kernel.
    )doc";

    bind_registered_operation(
        module,
        ttnn::experimental::deinterleave_to_batch,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("input_height"),
            py::arg("input_width"),
            py::arg("stride_hw") = std::array<uint32_t, 2>{2, 2},
            py::arg("barrier_threshold") = 0,
            py::arg("compute_kernel_config") = std::nullopt,
        });

    bind_registered_operation(
        module,
        ttnn::experimental::deinterleave_local,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("input_height"),
            py::arg("input_width"),
            py::arg("stride_hw") = std::array<uint32_t, 2>{2, 2},
            py::arg("barrier_threshold") = 0,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::experimental::deinterleave
