// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>  // split returns a vector

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "split.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_split(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Splits `input_tensor` into chunks along dimension `dim`.

        With an int `split_size`, every chunk has `split_size` elements along `dim` except the last, which holds the
        rest, so there are ``ceil(input_tensor.shape[dim] / split_size)`` chunks. With a list, chunk ``i`` has
        ``split_size[i]`` elements along `dim`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device, with a size of at least 1 along `dim`. A TILE input must use 32x32 tiles.
            split_size (int or List[int]): the size of each chunk, or a list with the size of each chunk. Sizes must be at least 1, and the sizes in a list must add up to ``input_tensor.shape[dim]``.
            dim (int, optional): the dimension to split along, in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension. Defaults to `0`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the outputs. Defaults to `None`, which keeps the memory configuration of `input_tensor`, except that an L1 `input_tensor` split into more than 2 chunks gives DRAM outputs when it is too large for the free L1 of a single core.

        Returns:
            List[ttnn.Tensor]: the chunks, in order along `dim`, with the dtype and layout of `input_tensor`.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - TILE, ROW_MAJOR
               * - UINT8
                 - ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            ``BFLOAT8_B`` tensors need a rank of 2 or more and, when `dim` is one of the last two dimensions, every
            chunk but the last must have a size that is a multiple of 32.

            ``BFLOAT16`` TILE input values must not be NaN when the input has rank 1, or when `dim` is one of the last
            two dimensions and a chunk other than the last has a size that is not a multiple of 32.

        Memory Support:
            - Interleaved: DRAM and L1
        )doc";

    ttnn::bind_function<"split">(
        mod,
        doc,

        // Overload 1: single split_size (int64_t)
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, const std::optional<ttnn::MemoryConfig>&>(
                &ttnn::split),
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        // Overload 2: list of split_sizes (ttsl::SmallVector<int64_t>)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int64_t>&,
                int64_t,
                const std::optional<ttnn::MemoryConfig>&>(&ttnn::split),
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}
}  // namespace ttnn::operations::data_movement::detail
