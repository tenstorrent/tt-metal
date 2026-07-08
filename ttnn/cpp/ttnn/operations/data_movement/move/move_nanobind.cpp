// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "move.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_move(nb::module_& mod) {
    const auto* doc = R"doc(
            Moves the elements of the input tensor ``arg0`` to a location in memory with specified memory layout.

            If no memory layout is specified, output memory will be the same as the input tensor memory config.

            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | Argument | Description                | Data type                  | Valid range                     | Required |
            +==========+============================+============================+=================================+==========+
            | arg0     | Tensor to move             | Tensor                     | Tensor of shape [W, Z, Y, X]    | Yes      |
            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | arg1     | MemoryConfig of tensor of  | tt_lib.tensor.MemoryConfig | Default is same as input tensor | No       |
            |          | TT accelerator device      |                            |                                 |          |
            +----------+----------------------------+----------------------------+---------------------------------+----------+

            ``implementation`` selects the backing prim: ``"auto"`` (default) routes in-scope calls
            to the codegen prim and everything else to native; ``"native"`` always uses the
            existing native prim; ``"codegen"`` forces the codegen prim and raises if the call is
            out of scope.
        )doc";

    ttnn::bind_function<"move">(
        mod,
        doc,
        &ttnn::move,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("implementation") = "auto");
}

}  // namespace ttnn::operations::data_movement::detail
