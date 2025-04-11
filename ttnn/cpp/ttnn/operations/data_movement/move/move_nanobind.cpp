// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "move.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_move(nb::module_& mod) {
    auto doc = R"doc(
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
        )doc";

    bind_registered_operation(
        mod,
        ttnn::move,
        doc,
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::move)& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) { return self(input_tensor, memory_config); },
            nb::arg("input_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::data_movement::detail
