// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "generic_op.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::generic {

void bind_generic_operation(nb::module_& mod) {
    std::string doc =
        R"doc(
        Executes a custom operation with user-defined kernels on the device.

        The generic_op provides a flexible interface for constructing and executing custom operations
        on device hardware. It allows specifying custom compute kernels, data movement, and control flow.

        Args:
            io_tensors (List[ttnn.Tensor]): List of input tensors and output tensor(s). Output tensor(s) must be
                pre-allocated and included as the last element(s) in the list.
            program_descriptor (ttnn.ProgramDescriptor): Descriptor containing kernel specifications,
                computational buffer configurations, semaphores, and other execution parameters.

        Returns:
            ttnn.Tensor: Handle to the output tensor.

        Example:
            Refer to tests/ttnn/unit_tests/operations/test_generic_op.py for usage examples
        )doc";

    bind_registered_operation(
        mod, ttnn::generic_op, doc, ttnn::nanobind_arguments_t{nb::arg("io_tensors"), nb::arg("program_descriptors")});
}

}  // namespace ttnn::operations::generic
