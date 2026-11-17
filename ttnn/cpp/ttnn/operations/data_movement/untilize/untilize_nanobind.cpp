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
    // The BFLOAT16 NaN rule in this docstring tracks tenstorrent/tt-metal#31406; drop it once that issue is fixed.
    const auto* doc =
        R"doc(
            Changes the layout of the input tensor from TILE to ROW_MAJOR.

            The output has the input's logical shape: the tile padding of a tensor whose last two dimensions are not
            multiples of 32 is left out.

            Args:
                input_tensor (ttnn.Tensor): the input tensor, on device, in TILE layout with 32x32 tiles.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the input tensor's memory configuration. A BLOCK_SHARDED output must be in L1. When an interleaved input has tile padding, the operation can derive a sharded output's shard shape from the output shape instead of taking it from ``memory_config``.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Cores to run the operation on. Defaults to `None`, which lets the operation choose its cores. It needs an interleaved input and output and ``use_multicore=True``. When the last two dimensions are multiples of 32, the input must also be one tile row tall: the product of all its dimensions but the last must be 32.

            Returns:
                ttnn.Tensor: the output tensor, in ROW_MAJOR layout with the input's dtype. A ``BFLOAT8_B`` input gives a ``BFLOAT16`` output.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, BFLOAT8_B, INT32, UINT32, UINT16, UINT8
                     - TILE

                ``BFLOAT16`` input values must not be NaN.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded input: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED and WIDTH_SHARDED in DRAM
                - Sharded output: HEIGHT_SHARDED and WIDTH_SHARDED in L1 or DRAM; BLOCK_SHARDED in L1
                - ND sharded: L1 input and output; an ND sharded output from an interleaved input needs the last two dimensions to be multiples of 32
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
        nb::arg("runtime_rows") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native untilize prim unconditionally. Not part of the ttnn API;
            use ttnn.untilize, which selects a prim on its own. `runtime_rows` makes the multi-core
            interleaved program derive its per-core rows on device, so one program serves every height.
            It has no effect on the other prims (a padded input goes through untilize_with_unpadding, a
            sharded input and a height the single-core / L1 rules route elsewhere keep their own programs),
            and a rank-4 input whose upper dims change still compiles per shape: extra programs, never a
            wrong result.
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
