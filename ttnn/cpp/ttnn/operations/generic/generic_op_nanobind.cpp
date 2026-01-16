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
            program_descriptor (ttnn.ProgramDescriptor or ttnn.MeshProgramDescriptor): Descriptor containing
                kernel specifications, computational buffer configurations, semaphores, and other execution
                parameters. Use ProgramDescriptor for SPMD mode (same program on all devices) or
                MeshProgramDescriptor for explicit per-device control.

        Returns:
            ttnn.Tensor: Handle to the output tensor.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op.py for usage examples
        )doc";

    // Use concrete type for self
    using registered_operation_t = std::decay_t<decltype(ttnn::generic_op)>;

    auto mesh_program_invoke = [](const registered_operation_t& self,
                                  const std::vector<Tensor>& io_tensors,
                                  const tt::tt_metal::experimental::MeshProgramDescriptor& desc) {
        return self(io_tensors, desc);
    };

    auto program_invoke = [](const registered_operation_t& self,
                             const std::vector<Tensor>& io_tensors,
                             const tt::tt_metal::ProgramDescriptor& desc) { return self(io_tensors, desc); };

    bind_registered_operation(
        mod,
        ttnn::generic_op,
        doc,
        // Overload for MeshProgramDescriptor (explicit mesh control)
        ttnn::nanobind_overload_t{mesh_program_invoke, nb::arg("io_tensors"), nb::arg("mesh_program_descriptor")},
        // Overload for ProgramDescriptor (SPMD mode - broadcasts to all devices)
        ttnn::nanobind_overload_t{program_invoke, nb::arg("io_tensors"), nb::arg("program_descriptor")});
}

}  // namespace ttnn::operations::generic
