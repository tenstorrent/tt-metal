// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <string>
#include <vector>

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
        Compiles and finalizes the program that a generic_op call with the same arguments would enqueue, without
        enqueueing it: ``ttnn.generic_op`` for a ProgramDescriptor, ``ttnn.experimental.generic_op`` for a
        ProgramSpec.

        Use it to find out whether a program compiles and fits the device's kernel-configuration buffer before
        committing resources to it. The prepared workload is inserted into the program cache, so the next
        generic_op call with the same tensors and program is a cache hit. A failed preparation leaves the program
        cache unchanged, with one exception: when a ProgramSpec is already cached, its run args are applied to the
        cached program first, so DFB size overrides stay in effect even if the preparation then fails.

        Args:
            io_tensors (List[ttnn.Tensor]): The tensors the generic_op call would receive.
            program_descriptor (ttnn.ProgramDescriptor): The program to prepare in SPMD mode (same program on all
                devices).
            mesh_program_descriptor (ttnn.MeshProgramDescriptor): The per-device programs to prepare, for explicit
                per-device control. Pass exactly one of ``program_descriptor`` and ``mesh_program_descriptor``, or
                pass ``spec`` and ``run_args`` instead.
            spec (ttnn.experimental.ProgramSpec or ttnn.experimental.MeshProgramSpec): The Metal 2.0 program, as for
                ``ttnn.experimental.generic_op``.
            run_args (ttnn.experimental.ProgramRunArgs or ttnn.experimental.MeshProgramRunArgs): Its run args, with
                real tensors.

        Raises:
            RuntimeError: If a kernel does not compile, if the program configuration does not fit the
                kernel-configuration buffer, or if the tensors' MeshDevice has no local devices. With a ProgramSpec,
                also for each error ``ttnn.experimental.generic_op`` raises.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op_preparation.py and
            test_generic_op_program_spec.py for usage examples
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

    // The Metal 2.0 forms join this binding: a second prepare_generic_op in the module would replace the first.
    auto program_spec_overload = ttnn::overload_t(
        static_cast<void (*)(
            const std::vector<Tensor>&,
            const ttnn::experimental::ProgramSpec&,
            const ttnn::experimental::ProgramRunArgs&)>(&ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    auto mesh_program_spec_overload = ttnn::overload_t(
        static_cast<void (*)(
            const std::vector<Tensor>&,
            const ttnn::experimental::MeshProgramSpec&,
            const ttnn::experimental::MeshProgramRunArgs&)>(&ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    ttnn::bind_function<"prepare_generic_op", "ttnn.experimental.">(
        mod, doc, mesh_program_overload, program_overload, program_spec_overload, mesh_program_spec_overload);
}

void bind_generic_operation_program_spec(nb::module_& mod) {
    using ttnn::experimental::MeshProgramRunArgs;
    using ttnn::experimental::MeshProgramSpec;
    using ttnn::experimental::ProgramRunArgs;
    using ttnn::experimental::ProgramSpec;

    const char* doc = R"doc(
        Runs a Metal 2.0 program, given as a ttnn.experimental.ProgramSpec (everything fixed about the program) and
        a ttnn.experimental.ProgramRunArgs (the values of this call), with named arguments and tensors bound by name.
        Experimental: the API may still change.

        The first call with a given program compiles its kernels; later calls with the same program write only the
        new argument values. Every call checks the run args against the spec: each kernel gets exactly its declared
        arguments, each array with its declared length, each tensor parameter gets a tensor from io_tensors that
        matches its spec, and each DFB size override names a DFB of the spec, at most once and with no size of 0.
        Whether every core got its per-core values is checked on later calls only when
        ttnn.CONFIG.validate_program_args is on; with it off, a core that a later call leaves out keeps the values it
        was given before.

        Args:
            io_tensors (List[ttnn.Tensor]): The tensors the program uses, with the preallocated output last. Each
                tensor parameter binds its own entry, so a tensor bound to two parameters is listed twice. The list
                may hold only the output.
            spec (ttnn.experimental.ProgramSpec or ttnn.experimental.MeshProgramSpec): The program, run on the
                devices of the tensor in io_tensors that is on the fewest devices (every other tensor must be on them
                too), or a program for each mesh coordinate range.
            run_args (ttnn.experimental.ProgramRunArgs or ttnn.experimental.MeshProgramRunArgs): The values of this
                call, or of each range's program.

        Returns:
            ttnn.Tensor: The output tensor, the last entry of io_tensors. When the tensors are not all on the
                same devices, only its parts on the devices of the tensor in io_tensors that is on the fewest
                devices.

        Raises:
            RuntimeError: If the run args do not match the spec, naming the kernel, argument, tensor parameter or
                DFB at fault; if a tensor is on another MeshDevice or misses a device the program runs on; if the
                MeshProgramSpec has no programs, its ranges overlap or reach outside the mesh, or the run args' ranges
                differ from the spec's; if io_tensors is empty or holds a tensor that is not allocated on a device, or
                one allocated per core; if Metal rejects the run args, such as a DFB size it cannot take; or, on a
                first call, if Metal rejects the spec, a kernel does not compile or the program does not fit the
                kernel-configuration buffer.

        Example:
            See the TT-NN guide "Running your own kernels with generic_op and a ProgramSpec", and
            tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py for more examples.
        )doc";

    auto program_spec_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const ProgramSpec&, const ProgramRunArgs&)>(
            &ttnn::experimental::generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    auto mesh_program_spec_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const MeshProgramSpec&, const MeshProgramRunArgs&)>(
            &ttnn::experimental::generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    ttnn::bind_function<"generic_op", "ttnn.experimental.">(
        mod, doc, program_spec_overload, mesh_program_spec_overload);

    mod.def(
        "compute_program_spec_hash",
        &ttnn::experimental::compute_program_spec_hash,
        nb::arg("spec"),
        R"doc(
        The hash of a ttnn.experimental.ProgramSpec that ttnn.experimental.generic_op keys its program cache on,
        combined with each program's mesh range. Runtime argument values, ProgramSpec.name and WorkUnitSpec.name are
        not part of it, and tensor parameter specs are hashed under their relaxations.
        )doc");
}

}  // namespace ttnn::operations::generic
