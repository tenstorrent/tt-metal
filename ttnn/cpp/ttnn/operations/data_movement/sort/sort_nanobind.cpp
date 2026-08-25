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
            stable (bool, optional): If `True`, ensures the original order of equal elements is preserved. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (tuple of ttnn.Tensor, optional): Preallocated output tensors for the sorted values and indices. Defaults to `None`. The index tensor must be of type uint16 or uint32.

        Returns:
            List of ttnn.Tensor: A list containing two tensors: The first tensor contains the sorted values, the second tensor contains the indices of the original elements in the sorted order.

        Additional info:

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

            Index dtype is UINT16 by default and auto-promoted to UINT32 when any of the
            following holds: the sort dim is >= 65535; the input dtype is FLOAT32 or UINT16;
            or stable=True with a BFLOAT16 input at padded sort width <= 2048 (any device) or
            at power-of-two padded sort width 2048..65536 on Blackhole (the mergesort row
            engine; widths 512/1024 are raised to 2048 by an internal pad rider). The
            Blackhole stable-BFLOAT16 promotion at padded widths 8192..32768 is a behavior
            change from earlier releases, which returned UINT16 there. Preallocating UINT16
            index tensors for a stable sort opts out to the comparator engines and keeps
            UINT16. Unstable BFLOAT16 sorts keep UINT16 at every width. The composite early
            exits (scalar, dim-size-1, and zero-size tensors, which return empty tensors
            matching torch.sort) follow the same dtype rule.

            NaN input is unsupported (undefined ordering) for BFLOAT16: the bfloat16 datapath
            canonicalizes NaN to same-sign infinity before comparing, so NaN placement deviates
            from torch.sort's NaN-last ordering. Mask or replace NaNs before sorting.

            With stable=False, the returned indices always gather the sorted values from the
            input but need not form a permutation within tie groups on the CrossCore path
            (duplicate indices may appear inside a tie group; see issue #54043). On Blackhole,
            BFLOAT16 sorts of power-of-two padded width 512..65536 are served by the mergesort
            row engine, whose unstable output IS the torch-stable permutation — #54043 is
            unreachable there. The residue remains for FLOAT32 unstable widths on the CrossCore
            path and for non-Blackhole devices. Use stable=True when a true permutation is
            required.

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
