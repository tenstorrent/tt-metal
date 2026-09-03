// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "sort_nanobind.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "sort.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_sort_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Sorts the elements of the input tensor along the specified dimension in ascending order by default.
        If no dimension is specified, the last dimension of the input tensor is used.

        Args:
            input_tensor (ttnn.Tensor): The input tensor to be sorted.

        Keyword Arguments:
            dim (int, optional): The dimension along which to sort. Defaults to `-1` (last dimension).
            descending (bool, optional): If `True`, sorts in descending order. Defaults to `False`.
            stable (bool, optional): If `True`, ensures the original order of equal elements is preserved. Defaults to `False`. With `stable=False` the returned indices are a valid permutation (no duplicates inside tie groups, no out-of-range indices), but which of several equal elements comes first is unspecified. Known exception: on the MultiCore DRAM factory, wide `float32` descending sorts can still emit padding indices past the logical row (issue #53326). On the multi-core cross-core path, `float32` inputs have `-0.0` canonicalized to `+0.0` in the returned values for both stabilities.
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (tuple of ttnn.Tensor, optional): Preallocated output tensors for the sorted values and indices. Defaults to `None`. The index tensor must be of type uint16 or uint32.

        Returns:
            List of ttnn.Tensor: A list containing two tensors: The first tensor contains the sorted values, the second tensor contains the indices of the original elements in the sorted order.

        Note:

            Supported dtypes and layouts for input tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16
                  - TILE, ROW_MAJOR
                * - UINT16
                  - TILE, ROW_MAJOR
                * - FLOAT32
                  - TILE, ROW_MAJOR

            Supported dtypes and layouts for index tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - UINT16, UINT32
                  - TILE, ROW_MAJOR

            NaN input is unsupported (undefined ordering) for BFLOAT16: the bfloat16 datapath
            canonicalizes NaN to same-sign infinity before comparing, so NaN placement deviates
            from torch.sort's NaN-last ordering. Mask or replace NaNs before sorting.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED (DRAM and L1)
    )doc";

    ttnn::bind_function<"sort">(
        mod,
        doc,
        &ttnn::sort,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim") = -1,
        nb::arg("descending") = false,
        nb::arg("stable") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("out") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
