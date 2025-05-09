// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "move.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_move(pybind11::module& module) {
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
        module,
        ttnn::move,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::move)& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, memory_config); },
            pybind11::arg("input_tensor").noconvert(),
            pybind11::kw_only(),
            pybind11::arg("memory_config") = std::nullopt,
            pybind11::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::data_movement::detail
