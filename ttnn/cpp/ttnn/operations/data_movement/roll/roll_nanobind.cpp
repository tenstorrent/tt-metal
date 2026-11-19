// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

#include "roll_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/roll/roll.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_roll(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Rolls the elements of `input_tensor` along the given dimensions, as ``torch.roll`` does: each element moves
        `shifts` places along its dimension, and the elements that pass the end of the dimension come back at its start.

        `shifts` and `dim` are both ints, or both lists of the same length that pair each shift with a dimension. A
        second form, ``ttnn.roll(input_tensor, shifts)`` with an int `shifts` and no `dim`, rolls the tensor as if
        flattened, then gives it back its shape; the input and the output must then be interleaved.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            shifts (int or List[int]): the number of places the elements move along each dimension in `dim`. Negative values move them toward the start of the dimension. A value can be larger than the size of its dimension.
            dim (int or List[int]): the dimensions to roll. Negative values count from the last dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which keeps the input's memory configuration.

        Returns:
            ttnn.Tensor: the rolled tensor, with the shape, dtype and layout of `input_tensor`.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - UINT8
                 - ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            TILE tensors must use the standard 32x32 tile.

            ``BFLOAT8_B`` tensors need `dim`, and each rolled dimension among the last two must have a size and a shift
            that are multiples of 32.

            ``UINT8`` tensors need `dim`.

            ``BFLOAT16`` input values must not be NaN when `dim` is omitted or includes one of the last two dimensions.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1. The shard grid must be a single rectangular core range. When a dimension other than the last is rolled, the shard height must be at most the product of the sizes of the dimensions from the rolled one through the second-to-last.
        )doc";

    ttnn::bind_function<"roll">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int>&,
                const ttsl::SmallVector<int>&,
                const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int, int, const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int, const std::optional<MemoryConfig>&>(&ttnn::roll),
            nb::arg("input_tensor"),
            nb::arg("shifts"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
