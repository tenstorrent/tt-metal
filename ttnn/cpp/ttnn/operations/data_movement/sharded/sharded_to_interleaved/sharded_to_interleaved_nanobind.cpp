// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "sharded_to_interleaved.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor sharded_to_interleaved_wrapper(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType>& output_dtype) {
    return ttnn::sharded_to_interleaved(
        input_tensor, memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG), output_dtype);
}

void bind_sharded_to_interleaved(nb::module_& mod) {
    const auto* doc = R"doc(
        Copies a sharded tensor into an interleaved tensor with the memory configuration `memory_config`. The output
        has the input's shape, layout and values. `output_dtype` converts a TILE input to another dtype.

        Args:
            input_tensor (ttnn.Tensor): the sharded input tensor, see Memory Support. A TILE input must have 32x32 tiles. For a ROW_MAJOR input, the shard width in bytes must be a multiple of 16.
            memory_config (ttnn.MemoryConfig, optional): interleaved memory configuration of the output, see Memory Support. Defaults to `None`, which gives an interleaved DRAM output.
            output_dtype (ttnn.DataType, optional): Data type of the output. Defaults to `None`, which keeps the input dtype. A TILE input of BFLOAT16, FLOAT32 or BFLOAT8_B can be converted to any of these three dtypes. Input values must not be NaN when the dtype changes.

        Returns:
            ttnn.Tensor: the interleaved output tensor, with the input's shape and layout.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

        Memory Support:
            - Input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1
            - Output: interleaved DRAM and L1
        )doc";

    ttnn::bind_function<"sharded_to_interleaved">(
        mod,
        doc,
        &sharded_to_interleaved_wrapper,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::data_movement
