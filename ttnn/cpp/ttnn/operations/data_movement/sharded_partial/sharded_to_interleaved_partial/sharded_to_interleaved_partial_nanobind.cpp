// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_partial_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "sharded_to_interleaved_partial.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_sharded_to_interleaved_partial(
    nb::module_& mod, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor").noconvert(),
            nb::arg("cache_tensor").noconvert(),
            nb::arg("num_slices"),
            nb::arg("slice_index"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_dtype") = nb::none(),
        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void bind_sharded_to_interleaved_partial(nb::module_& mod) {
    detail::bind_sharded_to_interleaved_partial(
        mod,
        ttnn::sharded_to_interleaved_partial,
        R"doc(

        Converts a partial tensor from sharded_to_interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`cache_tensor` (ttnn.Tensor): cache tensor
            * :attr:`num_slices` (int): Number of slices.
            * :attr:`slice_index` (int): Slice index.

        Keyword Args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, cache_tensor, 4, 2)

        )doc");
}

}  // namespace ttnn::operations::data_movement
