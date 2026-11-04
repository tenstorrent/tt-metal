// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "untilize.hpp"
#include "untilize_force.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_untilize(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Changes the layout of the input tensor from TILE to ROW_MAJOR.

            An interleaved input whose logical shape is not tile-aligned gives an output with the logical shape, without the
            tile padding.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, on device.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`, which uses the input tensor's memory configuration.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which uses the whole device. It needs interleaved input and output, ``use_multicore=True`` and a tensor one tile row tall.

            Returns:
                ttnn.Tensor: the output tensor, in ROW_MAJOR layout with the input's dtype. ``BFLOAT8_B`` input gives ``BFLOAT16`` output.

            Note:
                Supported dtypes and layouts for the input tensor:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, BFLOAT8_B, INT32, UINT32, UINT16, UINT8
                     - TILE

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input in L1 or DRAM; the output can be interleaved or sharded, except BLOCK_SHARDED in DRAM
                - ND sharded: input and output
        )doc";

    ttnn::bind_function<"untilize">(
        mod,
        doc,
        &ttnn::untilize,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("sub_core_grids") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See untilize_force.hpp.
    mod.def(
        "untilize_force_native",
        &untilize_force_native,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("use_multicore") = true,
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native untilize prim unconditionally. Not part of the ttnn API;
            use ttnn.untilize, which selects a prim on its own.
        )doc");

    mod.def(
        "untilize_force_codegen",
        &untilize_force_codegen,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen untilize prim unconditionally, raising for a case outside
            its support scope rather than falling back to native. Not part of the ttnn API; use
            ttnn.untilize, which selects a prim on its own.
        )doc");
}
}  // namespace ttnn::operations::data_movement::detail
