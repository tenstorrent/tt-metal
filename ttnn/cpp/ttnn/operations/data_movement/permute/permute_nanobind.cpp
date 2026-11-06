// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "permute.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_permute(nb::module_& mod) {
    const auto* doc = R"doc(
        Permutes the dimensions of the input tensor according to the specified permutation.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dims (List[int]): the new order of the dimensions. Its length must equal the rank of the tensor; negative values count from the last dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`: an interleaved input keeps its memory configuration, and for a sharded input the output's memory configuration is derived from the input's. A ROW_MAJOR WIDTH_SHARDED or BLOCK_SHARDED input whose last two dims are not both multiples of 32 can give an L1 interleaved output.
            pad_value (float, optional): value used to fill the tile padding that a permute exposes on ``BFLOAT16``, ``FLOAT32`` and ``UINT32`` TILE tensors. Defaults to `0.0`. ROW_MAJOR tensors ignore it.

        Returns:
            ttnn.Tensor: the output tensor, with the same dtype and layout as the input.

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

            ``BFLOAT8_B`` is supported for interleaved tensors of rank 4 or less.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output; DRAM-sharded input is also accepted
            - A sharded ROW_MAJOR output must have a shard width, in bytes, that is a multiple of the L1 alignment
    )doc";

    ttnn::bind_function<"permute">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int64_t>&,
                const std::optional<ttnn::MemoryConfig>&,
                float>(&ttnn::permute),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttsl::SmallVector<int64_t>&, float>(&ttnn::permute),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::arg("pad_value") = 0.0f));
}

}  // namespace ttnn::operations::data_movement::detail
