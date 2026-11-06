// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "repeat_interleave.hpp"
#include "repeat_interleave_force.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_repeat_interleave(nb::module_& mod) {
    const auto* doc = R"doc(
        Repeats each element of `input_tensor` `repeats` times along dimension `dim`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repeats (int): the number of times to repeat each element. Must be at least 1.
            dim (int): the dimension along which to repeat. Negative values count from the last dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`: an interleaved input, or any input with ``repeats=1``, keeps its memory configuration. For a sharded input with ``repeats`` > 1, see Memory Support. A given ``memory_config`` sets the output's memory configuration.

        Returns:
            ttnn.Tensor: the output tensor, with the same dtype and layout as the input.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded, with ``repeats`` > 1 and no ``memory_config``: a ROW_MAJOR rank-4 input that is HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED in L1 keeps the same sharding when a dimension other than the last is repeated. Repeating the last dimension gives an L1 output whose sharding depends on the shard geometry. HEIGHT_SHARDED and BLOCK_SHARDED inputs on this path need a device opened with a non-zero ``l1_small_size``.
            - With ``repeats`` > 1 and no ``memory_config``, TILE sharded inputs, DRAM-sharded inputs and sharded inputs of rank other than 4 are first converted to DRAM interleaved and give a DRAM interleaved output
    )doc";

    ttnn::bind_function<"repeat_interleave">(
        mod,
        doc,
        &ttnn::repeat_interleave,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See repeat_interleave_force.hpp.
    mod.def(
        "repeat_interleave_force_native",
        &repeat_interleave_force_native,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native repeat_interleave implementation unconditionally. Not
            part of the ttnn API; use ttnn.repeat_interleave, which selects an implementation on its
            own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeats (number): the number of repetitions for each element.
                dim (number): the dimension to expand with the repetitions.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc");

    mod.def(
        "repeat_interleave_force_codegen",
        &repeat_interleave_force_codegen,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen repeat_interleave implementation unconditionally,
            raising for a case outside its support scope rather than falling back to native. Not
            part of the ttnn API; use ttnn.repeat_interleave, which selects an implementation on its
            own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeats (number): the number of repetitions for each element.
                dim (number): the dimension to expand with the repetitions.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor.

            Raises:
                RuntimeError: the codegen path does not support this case.
        )doc");
}

}  // namespace ttnn::operations::data_movement::detail
