// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "sharded_to_interleaved.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_sharded_to_interleaved(
    nb::module_& mod, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType>& output_dtype,
               const std::optional<bool>& is_l1_aligned) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                    output_dtype,
                    is_l1_aligned);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("is_l1_aligned") = false,
        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void bind_sharded_to_interleaved(nb::module_& mod) {
    detail::bind_sharded_to_interleaved(
        mod,
        ttnn::sharded_to_interleaved,
        R"doc(sharded_to_interleaved(input_tensor: ttnn.Tensor,  memory_config: ttnn.MemoryConfig, *) -> ttnn.Tensor

        Converts a tensor from sharded to interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (ttnn.MemoryConfig): Memory configuration for the operation, must be Interleaved.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

        )doc");
}

}  // namespace ttnn::operations::data_movement
