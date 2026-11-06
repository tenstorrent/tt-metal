// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"  // for ttsl::SmallVector<uint32_t>

#include "repeat.hpp"
#include "repeat_force.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

void bind_repeat(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a tensor that repeats `input_tensor` along each dimension the number of times given in `repeat_dims`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repeat_dims (List[int]): the number of repetitions for each dimension. A list shorter than the tensor's rank is padded with 1s at the front, and a longer list adds leading dimensions to the output.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`: an interleaved input keeps its memory configuration. A sharded input keeps its memory layout and buffer type, with a shard spec derived for the output shape, when such a spec can be derived; otherwise, and for an empty output, the output is interleaved in the input's buffer type. When ``optional_output_tensor`` is given, the output uses its memory configuration.
            optional_output_tensor (ttnn.Tensor, optional): Preallocated output tensor, with the output's shape, dtype and layout. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor, with the same dtype and layout as the input.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            A 0 in ``repeat_dims`` returns an empty tensor.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output in L1; DRAM-sharded input is also accepted
            - ND sharded: DRAM and L1
    )doc";

    ttnn::bind_function<"repeat">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            const ttsl::SmallVector<uint32_t>&,
            const std::optional<MemoryConfig>&,
            const std::optional<ttnn::Tensor>&>(&ttnn::repeat),
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See repeat_force.hpp.
    mod.def(
        "repeat_force_native",
        &detail::repeat_force_native,
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native repeat implementation unconditionally. Not part of the
            ttnn API; use ttnn.repeat, which selects an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeat_dims (SmallVector): the number of repetitions for each dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor. Takes no preallocated output -- a forced leg is a
                reference for comparison, not a call path with the full public surface.
        )doc");

    mod.def(
        "repeat_force_codegen",
        &detail::repeat_force_codegen,
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen repeat implementation unconditionally, raising for a
            case outside its support scope rather than falling back to native. Not part of the ttnn
            API; use ttnn.repeat, which selects an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeat_dims (SmallVector): the number of repetitions for each dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor. Takes no preallocated output -- a forced leg is a
                reference for comparison, not a call path with the full public surface.

            Raises:
                RuntimeError: the codegen path does not support this case.
        )doc");
}

}  // namespace ttnn::operations::data_movement
