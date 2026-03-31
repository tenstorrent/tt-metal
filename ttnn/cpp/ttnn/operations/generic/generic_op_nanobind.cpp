// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/kernel_types.hpp>
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

    // --- ProgramCompileInfo bindings ---

    nb::class_<tt::tt_metal::detail::KernelBinaryMeta>(
        mod, "KernelBinaryMeta", "Binary metadata for a single kernel binary (one per RISC processor).")
        .def_ro(
            "processor_type",
            &tt::tt_metal::detail::KernelBinaryMeta::processor_type,
            "Processor type index (maps to BRISC=0, NCRISC=1, TRISC0=2, etc.)")
        .def_ro(
            "packed_size",
            &tt::tt_metal::detail::KernelBinaryMeta::packed_size,
            "Packed binary size in bytes (actual L1 footprint)");

    nb::class_<tt::tt_metal::detail::KernelMeta>(
        mod, "KernelMeta", "Metadata for a compiled kernel including binary sizes.")
        .def_prop_ro(
            "name", [](const tt::tt_metal::detail::KernelMeta& self) { return std::string(self.name); }, "Kernel name")
        .def_prop_ro(
            "source",
            [](const tt::tt_metal::detail::KernelMeta& self) { return std::string(self.source); },
            "Kernel source path")
        .def_ro(
            "binary_meta",
            &tt::tt_metal::detail::KernelMeta::binary_meta,
            "Per-processor binary metadata (size per RISC)");

    nb::class_<ProgramCompileInfo>(
        mod,
        "ProgramCompileInfo",
        R"pbdoc(
            Post-compilation metadata for a generic_op program.

            Contains the kernel config buffer layout (offsets relative to the
            config slot start) and per-kernel binary sizes. Only valid after
            ttnn.generic_op() has been called at least once with the same
            io_tensors and program_descriptor.
        )pbdoc")
        .def_ro("rta_offset", &ProgramCompileInfo::rta_offset, "Runtime args offset from config slot start")
        .def_ro("sem_offset", &ProgramCompileInfo::sem_offset, "Semaphore config offset")
        .def_ro("sem_size", &ProgramCompileInfo::sem_size, "Semaphore config size in bytes")
        .def_ro("cb_offset", &ProgramCompileInfo::cb_offset, "Circular buffer config offset")
        .def_ro("cb_size", &ProgramCompileInfo::cb_size, "Circular buffer config size in bytes")
        .def_ro("dfb_offset", &ProgramCompileInfo::dfb_offset, "Dataflow buffer offset")
        .def_ro("dfb_size", &ProgramCompileInfo::dfb_size, "Dataflow buffer size in bytes")
        .def_ro("local_cb_size", &ProgramCompileInfo::local_cb_size, "Local circular buffer data size")
        .def_ro(
            "kernel_text_offset",
            &ProgramCompileInfo::kernel_text_offset,
            "Kernel binary offset from config slot start")
        .def_ro("kernel_text_size", &ProgramCompileInfo::kernel_text_size, "Total kernel binary size in bytes")
        .def_ro(
            "program_config_sizes",
            &ProgramCompileInfo::program_config_sizes,
            "Total config slot size per programmable core type")
        .def_ro("kernel_metas", &ProgramCompileInfo::kernel_metas, "Per-kernel metadata including binary sizes");

    mod.def(
        "get_program_compile_info",
        &get_program_compile_info,
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        R"pbdoc(
            Query the kernel config layout of a compiled generic_op program.

            Returns offsets and sizes of all components in the kernel config
            buffer (runtime args, semaphores, CB configs, kernel binaries),
            plus per-kernel binary metadata.

            Must be called after ttnn.generic_op(io_tensors, program_descriptor)
            has executed at least once so the program is in the cache.
        )pbdoc");
}

}  // namespace ttnn::operations::generic
