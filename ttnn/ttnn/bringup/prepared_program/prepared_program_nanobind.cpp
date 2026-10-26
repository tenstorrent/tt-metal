// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

#include "prepared_program.hpp"

namespace nb = nanobind;

namespace ttnn::operations::bringup::prepared_program::detail {

void bind_prepared_program(nb::module_& mod) {
    nb::class_<Prepared>(mod, "PreparedProgram", R"doc(
        A generic_op program prepared once: the descriptor is kept, and each call patches only the runtime-arg slots
        holding an io tensor's address and the CBs an io tensor backs (prepared_program.hpp has the contract: every
        per-call tensor a kernel addresses is in io_tensors, at the same positions every call).

            p = ttnn.bringup.prepare_program(mesh_program_descriptor, io_tensors)   # once
            ttnn.bringup.run_prepared(p, io_tensors)                                  # every call
        )doc")
        .def_prop_ro("patch_count", &Prepared::patch_count)
        .def_prop_ro("n_io", [](const Prepared& p) { return p.n_io; })
        .def_prop_ro("hash", [](const Prepared& p) { return static_cast<uint64_t>(p.hash); });

    mod.def(
        "prepare_program",
        [](const tt::tt_metal::experimental::MeshProgramDescriptor& desc, const std::vector<Tensor>& io_tensors) {
            return std::const_pointer_cast<Prepared>(prepare(desc, io_tensors));
        },
        nb::arg("mesh_program_descriptor"),
        nb::arg("io_tensors"),
        "Prepare a MeshProgramDescriptor for patched dispatch (copies it once; records the patch slots).");
    mod.def(
        "run_prepared",
        [](const std::shared_ptr<Prepared>& p, const std::vector<Tensor>& io_tensors) { return run(p, io_tensors); },
        nb::arg("prepared"),
        nb::arg("io_tensors"),
        "Dispatch a prepared program on these io tensors (the output last); returns the output.");
}

}  // namespace ttnn::operations::bringup::prepared_program::detail
