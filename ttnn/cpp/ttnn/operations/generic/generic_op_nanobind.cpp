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
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

// Defined in generic_op_device_operation.cpp
tt::stl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);

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

    mod.def(
        "precompile_program_descriptor",
        [](tt::tt_metal::distributed::MeshDevice* device, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
            TT_FATAL(device != nullptr, "precompile_program_descriptor: device must not be null");
            const auto devices = device->get_devices();
            TT_FATAL(!devices.empty(), "precompile_program_descriptor: MeshDevice has no devices");
            // Build the program from the descriptor and JIT-compile its kernels WITHOUT
            // enqueueing it. CompileProgram populates the on-disk JIT cache (hash-keyed
            // per kernel); a later ttnn.generic_op with a structurally-equal descriptor
            // reads those binaries back as a warm cache hit. Compiling for devices[0] is
            // sufficient: a homogeneous mesh shares one build_key, so one compile warms
            // the cache for the whole mesh. No command queue / program-cache state is
            // touched, so this is safe to call concurrently for distinct descriptors.
            tt::tt_metal::Program program{program_descriptor};
            tt::tt_metal::detail::CompileProgram(devices.front(), program);
        },
        nb::arg("device"),
        nb::arg("program_descriptor"),
        nb::call_guard<nb::gil_scoped_release>(),
        R"pbdoc(
            JIT-compile a ProgramDescriptor's kernels up front, without enqueuing.

            Builds the program from the descriptor and compiles its kernels into the
            on-disk JIT cache (keyed per kernel hash). Does NOT enqueue to any command
            queue and does NOT touch the device program cache, so it is safe to call
            concurrently from multiple Python threads (the GIL is released for the
            duration of the C++ compile). A subsequent ttnn.generic_op with a
            structurally-equal descriptor then hits the warm cache.

            Args:
                device (ttnn.MeshDevice): device whose build environment to compile for.
                program_descriptor (ttnn.ProgramDescriptor): descriptor to compile.
        )pbdoc");
}

}  // namespace ttnn::operations::generic
