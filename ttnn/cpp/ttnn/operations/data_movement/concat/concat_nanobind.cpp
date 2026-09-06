// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "concat_force.hpp"
#include "codegen/concat_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_concat(nb::module_& mod) {
    const auto* doc = R"doc(

        Args:
            input_tensor (List of ttnn.Tensor): the input tensors.
            dim (number): the concatenating dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            groups (int, optional): When `groups` is set to a value greater than 1, the inputs are split into N `groups` partitions, and elements are interleaved from each group into the output tensor. Each group is processed independently, and elements from each group are concatenated in an alternating pattern based on the number of groups. This is useful for recombining grouped convolution outputs during residual concatenation. Defaults to `1`. Currently, groups > `1` is only supported for two height sharded input tensors.

        Keyword Args:
            sub_core_grids (ttnn.CoreRangeSet, optional): Sub-core grid to use for interleaved (L1 or DRAM) output tensors. If provided, the concatenation will run on the specified sub-core grid instead of the full compute grid. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"concat">(
        mod,
        doc,
        &ttnn::concat,
        nb::arg("tensors"),
        nb::arg("dim") = 0,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor").noconvert() = nb::none(),
        nb::arg("groups") = 1,
        nb::arg("sub_core_grids") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See concat_force.hpp.
    mod.def(
        "concat_force_native",
        &concat_force_native,
        nb::arg("tensors"),
        nb::arg("dim") = 0,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("groups") = 1,
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native concat implementation unconditionally. Not part of the
            ttnn API; use ttnn.concat, which selects an implementation on its own.

            Args:
                tensors (List of ttnn.Tensor): the input tensors.
                dim (number): the concatenating dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`.
                groups (int, optional): see ttnn.concat. Defaults to `1`.
                sub_core_grids (ttnn.CoreRangeSet, optional): see ttnn.concat. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor. Takes no preallocated output -- a forced leg is a
                reference for comparison, not a call path with the full public surface.
        )doc");

    mod.def(
        "concat_force_codegen",
        &concat_force_codegen,
        nb::arg("tensors"),
        nb::arg("dim") = 0,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen concat implementation unconditionally, raising for a
            case outside its support scope rather than falling back to native. Not part of the ttnn
            API; use ttnn.concat, which selects an implementation on its own.

            Args:
                tensors (List of ttnn.Tensor): the input tensors.
                dim (number): the concatenating dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor. Takes neither `groups` nor `sub_core_grids`: no
                generated builder honours either.
        )doc");

    // Exported so the routing tests exercise the real ceiling instead of a copy of it that
    // stops testing the boundary the moment the C++ value moves.
    mod.attr("CONCAT_MAX_NWAY_INPUTS") = ttnn::prim::kConcatMaxNwayInputs;
}

}  // namespace ttnn::operations::data_movement::detail
