// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "generic_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

// Defined in generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);

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

    // Overload for MeshProgramDescriptor (explicit mesh control)
    auto mesh_program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(
            const std::vector<Tensor>&, const tt::tt_metal::experimental::MeshProgramDescriptor&)>(&ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"));

    // Overload for ProgramDescriptor (SPMD mode - broadcasts to all devices)
    auto program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const tt::tt_metal::ProgramDescriptor&)>(
            &ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"));

    ttnn::bind_function<"generic_op">(mod, doc.c_str(), mesh_program_overload, program_overload);

    mod.def(
        "compute_program_descriptor_hash",
        &compute_program_descriptor_hash,
        nb::arg("program_descriptor"),
        R"pbdoc(
            Compute structural hash of a ProgramDescriptor.

            Hashes kernel sources, compile-time args, core ranges, CB structure,
            and semaphores. Excludes runtime arg values and buffer addresses,
            making it suitable as a cache key for structural equivalence.
        )pbdoc");
}

void bind_generic_operation_preparation(nb::module_& mod) {
    const char* doc =
        R"doc(
        Compiles and finalizes the program that ``ttnn.generic_op`` would enqueue, without enqueueing it.

        Use it to find out whether a program compiles and fits the device's kernel-configuration buffer before
        committing resources to it. The prepared workload is inserted into the program cache, so the next
        ``ttnn.generic_op`` with the same tensors and descriptor is a cache hit. A failed preparation leaves the
        program cache unchanged.

        Args:
            io_tensors (List[ttnn.Tensor]): The input and output tensors ``ttnn.generic_op`` would receive.
            program_descriptor (ttnn.ProgramDescriptor): The program to prepare in SPMD mode (same program on all
                devices).
            mesh_program_descriptor (ttnn.MeshProgramDescriptor): The per-device programs to prepare, for explicit
                per-device control. Pass exactly one of ``program_descriptor`` and ``mesh_program_descriptor``.

        Raises:
            RuntimeError: If a kernel does not compile, if the program configuration does not fit the
                kernel-configuration buffer, or if the tensors' MeshDevice has no local devices.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op_preparation.py for usage examples
        )doc";

    auto mesh_program_overload = ttnn::overload_t(
        static_cast<void (*)(const std::vector<Tensor>&, const tt::tt_metal::experimental::MeshProgramDescriptor&)>(
            &ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"));

    auto program_overload = ttnn::overload_t(
        static_cast<void (*)(const std::vector<Tensor>&, const tt::tt_metal::ProgramDescriptor&)>(
            &ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"));

    ttnn::bind_function<"prepare_generic_op", "ttnn.experimental.">(mod, doc, mesh_program_overload, program_overload);
}

}  // namespace ttnn::operations::generic
