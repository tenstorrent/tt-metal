// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_pybind.hpp"
#include "generic_op.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::generic {

void bind_generic_operation(py::module& module) {
    std::string doc =
        R"doc(
        Generates a tensor to draw binary random numbers (0 or 1) from a Bernoulli distribution.
        Args:
            io_tensors (ttnn.Tensor): List of input tensors and output tensor. Output tensor must be the last element.
            program_descriptor
        Returns:
            ttnn.Tensor: handle to the output tensor.
        Example:
            >>> input = ttnn.to_device(ttnn.from_torch(torch.empty(3, 3).uniform_(0, 1), dtype=torch.bfloat16)), device=device)
            >>> program_descriptor = ...
            >>> output = ttnn.generic_op(io_tensors, program_descriptor)
        )doc";

    bind_registered_operation(
        module, ttnn::generic_op, doc, ttnn::pybind_arguments_t{py::arg("io_tensors"), py::arg("program_descriptors")});
}

}  // namespace ttnn::operations::generic
