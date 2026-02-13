// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_partial_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "sharded_to_interleaved_partial.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_sharded_to_interleaved_partial(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a partial tensor from sharded_to_interleaved memory layout

        Args:
            input_tensor (ttnn.Tensor): input tensor
            cache_tensor (ttnn.Tensor): cache tensor
            num_slices (int): Number of slices.
            slice_index (int): Slice index.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the operation. Defaults to `None`.
            output_dtype (Optional[ttnn.DataType]): Output data type, defaults to same as input. Defaults to `None`.

        Returns:
            ttnn.Tensor: the cache tensor with the partial data written.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved_partial(tensor, cache_tensor, 4, 2)

        )doc";

    ttnn::bind_function<"sharded_to_interleaved_partial">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::sharded_to_interleaved_partial,
            nb::arg("input_tensor").noconvert(),
            nb::arg("cache_tensor").noconvert(),
            nb::arg("num_slices"),
            nb::arg("slice_index"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_dtype") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
