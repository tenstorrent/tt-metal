// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "sharded_to_interleaved.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_sharded_to_interleaved(nb::module_& mod) {
    const auto* doc = R"doc(
        Converts a tensor from sharded to interleaved memory layout

        Args:
            input_tensor (ttnn.Tensor): input tensor
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation, must be Interleaved.

        Keyword Args:
            output_dtype (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

        )doc";

    ttnn::bind_function<"sharded_to_interleaved">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const MemoryConfig&, const std::optional<DataType>&>(
                &ttnn::sharded_to_interleaved),
            nb::arg("input_tensor").noconvert(),
            nb::arg("memory_config"),
            nb::kw_only(),
            nb::arg("output_dtype") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
