// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"

#include "expand.hpp"

namespace ttnn::operations::data_movement {

namespace {

ttnn::Tensor expand_wrapper(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& output_shape,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    return ttnn::expand(input_tensor, output_shape, memory_config);
}

}  // namespace

void bind_expand(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a tensor whose dimensions of size 1 in `input_tensor` are expanded to the sizes in `output_shape`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            output_shape (List[int]): the shape of the output, with at least as many dimensions as `input_tensor`. Its last dimensions pair with the dimensions of `input_tensor`: a dimension of size 1 can take any size of 0 or more, and every other dimension must keep its size; -1 keeps the size of a dimension. Leading dimensions that `input_tensor` does not have are added with the given sizes, which must be 0 or more. A size of 0 returns an empty tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`: an interleaved input keeps its memory configuration. A sharded input keeps its memory layout and buffer type, with a shard spec derived for the output shape, when such a spec can be derived; otherwise, and for an empty output, the output is interleaved in the input's buffer type.

        Returns:
            ttnn.Tensor: the output tensor, with the same dtype and layout as the input.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            For a ``BFLOAT16`` TILE input whose last two dims are not both multiples of 32, input values must not be NaN.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output in L1; DRAM-sharded input is also accepted with an interleaved ``memory_config``
        )doc";

    ttnn::bind_function<"expand">(
        mod,
        doc,
        &expand_wrapper,
        nb::arg("input_tensor"),
        nb::arg("output_shape"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
