// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "sharded_to_interleaved.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::data_movement {

void bind_sharded_to_interleaved(nb::module_& mod) {
    const char* doc = R"doc(
        Converts a tensor from sharded to interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (ttnn.MemoryConfig): Memory configuration for the operation, must be Interleaved.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

        )doc";

    // Wrapper to handle optional memory_config with default
    mod.def(
        "sharded_to_interleaved",
        [](const ttnn::Tensor& input_tensor,
           const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
           const std::optional<tt::tt_metal::DataType>& output_dtype) -> ttnn::Tensor {
            return ttnn::sharded_to_interleaved(
                input_tensor,
                memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                output_dtype);
        },
        doc,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::data_movement
