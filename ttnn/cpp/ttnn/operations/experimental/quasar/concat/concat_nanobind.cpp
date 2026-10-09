// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "concat.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_concat(nb::module_& mod) {
    const auto* doc = R"doc(
        Quasar (Metal 2.0) port of ``ttnn.concat``: concatenates the input tensors along ``dim``.

        Args:
            input_tensor (List of ttnn.Tensor): the input tensors.
            dim (number): the concatenating dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to DRAM interleaved, like ``ttnn.concat``.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Not supported; must be `None`.
            groups (int, optional): Must be `1`. Grouped concat exists only in ``ttnn.concat``'s L1 height-sharded factory, which is not ported to Quasar.
            sub_core_grids (ttnn.CoreRangeSet, optional): Sub-core grid to use for interleaved (L1 or DRAM) output tensors. If provided, the concatenation will run on the specified sub-core grid instead of the full compute grid. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"concat", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::concat,
        nb::arg("tensors"),
        nb::arg("dim") = 0,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor").noconvert() = nb::none(),
        nb::arg("groups") = 1,
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
