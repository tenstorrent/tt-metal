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
        Returns a new tensor where singleton dimensions are expanded to a larger size.

        Unlike :func:`torch.expand`, this function is not zero-cost: ``ttnn.Tensor`` has no strided views, so the
        expanded tensor is materialized with :func:`ttnn.repeat`.

        As in :func:`torch.expand`, ``output_shape`` is aligned with the input shape from the last dimension:

        * Only dimensions of size 1 can be expanded.
        * ``-1``, or the input's size, keeps a dimension unchanged.
        * ``output_shape`` can have more dimensions than the input. The new leading dimensions must be given as non-negative sizes, not ``-1``. A size of 0 gives an empty tensor.

        Args:
            input_tensor (ttnn.Tensor): the tensor to expand, on device.
            output_shape (List[int]): the desired output shape. Its rank must be at least the rank of ``input_tensor``.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to `None`, which derives it from the input as :func:`ttnn.repeat` does.

        Returns:
            ttnn.Tensor: the expanded tensor, with shape ``output_shape`` and the same dtype and layout as the input. It is a copy, except that when no dimension is expanded and the memory configuration does not change, the input's data is reused without a copy.

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
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output in L1
            - DRAM sharded: HEIGHT_SHARDED and WIDTH_SHARDED input. A TILE input keeps its DRAM sharding by default or takes an interleaved ``memory_config``; a ROW_MAJOR input needs an interleaved ``memory_config``
            - ND sharded: input in DRAM or L1; by default the output is interleaved, in the input's buffer type
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
