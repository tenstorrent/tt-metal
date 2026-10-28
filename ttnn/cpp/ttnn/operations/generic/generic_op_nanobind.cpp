// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "generic_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/program_spec.hpp"
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
        cache unchanged.

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
            custom_program_hash (int, optional): As for ``ttnn.experimental.generic_op``. Defaults to None.

        Raises:
            RuntimeError: If a kernel does not compile, if the program configuration does not fit the
                kernel-configuration buffer, or if the tensors' MeshDevice has no local devices.

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
            const ttnn::experimental::ProgramRunArgs&,
            std::optional<uint64_t>)>(&ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"),
        nb::arg("custom_program_hash") = nb::none());

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
    using ttnn::program_spec::TENSOR_ARGS_ATTRIBUTE;

    nb::class_<MeshProgramSpec>(
        mod,
        "MeshProgramSpec",
        R"doc(
        Everything fixed about a mesh program: a ttnn.experimental.ProgramSpec for each mesh coordinate range, and
        an optional custom_program_hash that replaces the hash of the specs in the program cache key (which still
        includes the names the run args refer to, each kernel's cores and its vararg counts). The ranges must not
        overlap.
        Lists are copied in and out.
        )doc")
        .def(
            "__init__",
            [](MeshProgramSpec* self,
               std::vector<std::pair<MeshCoordinateRange, ProgramSpec>> programs,
               std::optional<uint64_t> custom_program_hash) {
                new (self) MeshProgramSpec{.programs = std::move(programs), .custom_program_hash = custom_program_hash};
            },
            nb::kw_only(),
            nb::arg("programs") = std::vector<std::pair<MeshCoordinateRange, ProgramSpec>>{},
            nb::arg("custom_program_hash") = nb::none())
        .def_rw("programs", &MeshProgramSpec::programs, nb::rv_policy::copy)
        .def_rw("custom_program_hash", &MeshProgramSpec::custom_program_hash, nb::for_setter(nb::arg("value").none()));

    // A Python ProgramRunArgs keeps its tensors alive in an attribute. A MeshProgramRunArgs holds C++ copies of its
    // run args, so it keeps a copy of each of those attributes, in the order of its programs.
    const auto set_programs = [](nb::pointer_and_handle<MeshProgramRunArgs> self,
                                 const std::vector<std::pair<MeshCoordinateRange, nb::object>>& programs) {
        std::vector<std::pair<MeshCoordinateRange, ProgramRunArgs>> run_args;
        nb::list tensors;
        run_args.reserve(programs.size());
        for (const auto& [range, program_run_args] : programs) {
            run_args.emplace_back(range, nb::cast<const ProgramRunArgs&>(program_run_args));
            tensors.append(nb::getattr(program_run_args, TENSOR_ARGS_ATTRIBUTE, nb::dict()).attr("copy")());
        }
        self.p->programs = std::move(run_args);
        nb::setattr(self.h, TENSOR_ARGS_ATTRIBUTE, tensors);
    };
    nb::class_<MeshProgramRunArgs>(
        mod,
        "MeshProgramRunArgs",
        nb::dynamic_attr(),
        R"doc(
        The values of one mesh program run: a ttnn.experimental.ProgramRunArgs for each range of the
        MeshProgramSpec. It keeps the tensors of its run args alive. Lists are copied in and out.
        )doc")
        .def(
            "__init__",
            [set_programs](
                nb::pointer_and_handle<MeshProgramRunArgs> self,
                const std::vector<std::pair<MeshCoordinateRange, nb::object>>& programs) {
                new (self.p) MeshProgramRunArgs{};
                set_programs(self, programs);
            },
            nb::kw_only(),
            nb::arg("programs") = std::vector<std::pair<MeshCoordinateRange, nb::object>>{})
        .def(
            "__setattr__",
            [](nb::handle self, const nb::str& name, nb::handle value) {
                ttnn::program_spec::set_known_attribute(self, name, value, nb::type<MeshProgramRunArgs>());
            },
            nb::arg("name"),
            nb::arg("value").none())
        .def_prop_rw(
            "programs",
            [](nb::pointer_and_handle<MeshProgramRunArgs> self) {
                nb::object tensors = nb::getattr(self.h, TENSOR_ARGS_ATTRIBUTE);
                nb::list programs;
                for (std::size_t i = 0; i < self.p->programs.size(); ++i) {
                    const auto& [range, run_args] = self.p->programs[i];
                    nb::object run_args_copy = nb::cast(run_args, nb::rv_policy::copy);
                    nb::setattr(run_args_copy, TENSOR_ARGS_ATTRIBUTE, tensors[i].attr("copy")());
                    programs.append(nb::make_tuple(range, run_args_copy));
                }
                return programs;
            },
            set_programs);

    const char* doc = R"doc(
        Runs a Metal 2.0 program, given as a ttnn.experimental.ProgramSpec (everything fixed about the program) and
        a ttnn.experimental.ProgramRunArgs (the values of this call), with named arguments and tensors bound by name.
        Experimental: the API may still change.

        The first call with a given program compiles its kernels; later calls with the same program write only the
        new argument values. Every call checks the run args against the spec: each kernel gets exactly its declared
        arguments and varargs, and each tensor parameter gets a tensor from io_tensors that matches its spec.
        Whether every core got its per-core values is checked on later calls only when
        ttnn.CONFIG.validate_program_args is on.

        Args:
            io_tensors (List[ttnn.Tensor]): The tensors the program uses, with the preallocated output last. Each
                tensor parameter binds its own entry, so a tensor bound to two parameters is listed twice. The list
                may hold only the output.
            spec (ttnn.experimental.ProgramSpec or ttnn.experimental.MeshProgramSpec): The program, run on every
                device that holds the tensors, or a program for each mesh coordinate range.
            run_args (ttnn.experimental.ProgramRunArgs or ttnn.experimental.MeshProgramRunArgs): The values of this
                call, or of each range's program.
            custom_program_hash (int, optional): With a ProgramSpec only (a MeshProgramSpec has its own field):
                replaces the hash of the spec in the program cache key, which still includes the names the run args
                refer to, each kernel's cores and its vararg counts. A non-negative int below 2**64; keeping keys
                unique is then the caller's job. Defaults to None.

        Returns:
            ttnn.Tensor: The output tensor, the last entry of io_tensors.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py for usage examples
        )doc";

    auto program_spec_overload = ttnn::overload_t(
        static_cast<Tensor (*)(
            const std::vector<Tensor>&, const ProgramSpec&, const ProgramRunArgs&, std::optional<uint64_t>)>(
            &ttnn::experimental::generic_op),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"),
        nb::arg("custom_program_hash") = nb::none());

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
        The hash of a ttnn.experimental.ProgramSpec that ttnn.experimental.generic_op keys its program cache on when
        no custom_program_hash is given. Argument values, ProgramSpec.name and WorkUnitSpec.name are not part of it,
        and tensor parameter specs are hashed under their relaxations. generic_op computes it on every call, so a
        caller that runs one spec many times can compute it once and pass it as custom_program_hash.
        )doc");
}

}  // namespace ttnn::operations::generic
