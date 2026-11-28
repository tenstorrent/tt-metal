// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "clone.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::clone {
void bind_clone_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Clones the input tensor into a newly allocated tensor, optionally converting its data type and moving it to another memory configuration. The layout does not change.

        Args:
            input (ttnn.Tensor): the input tensor, allocated on device. A TILE input must use 32x32 tiles.

        Keyword Args:
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`, which keeps the input's data type.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to `None`, which uses the input tensor's memory configuration.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): the configuration for the compute kernel that converts the data type. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): cores to run on for interleaved tensors. Defaults to `None`, which uses all cores.

        Returns:
            ttnn.Tensor: a copy of the input, never a view. It has the input's logical and padded shape and layout, with the data type given by ``dtype`` and the memory configuration given by ``memory_config``.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            Converting the data type needs a TILE input. Converting to or from ``INT32`` is not supported: it is not
            rejected, but gives wrong values.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED, in DRAM or L1. The input and output must both be sharded, with the same memory layout and the same shard spec; only the buffer type can differ
            - A sharded input with an interleaved output, or the reverse, is not supported
    )doc";

    ttnn::bind_function<"clone">(
        mod,
        doc,
        &ttnn::clone,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}
}  // namespace ttnn::operations::data_movement::clone
