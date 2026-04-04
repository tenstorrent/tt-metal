// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "patchable_generic_op_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "device/patchable_generic_op_device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {

void bind_patchable_generic_op(nb::module_& mod) {
    mod.def(
        "patchable_generic_op",
        [](const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
            TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
            auto* mesh_device = io_tensors.front().device();
            TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

            tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
            // Copy: mesh workload uses this descriptor; must not require a mutable Python binding.
            mesh_program_descriptor.mesh_programs.emplace_back(
                ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

            (void)ttnn::prim::patchable_generic_op(io_tensors, mesh_program_descriptor);
        },
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        R"doc(
        Dispatch like ``generic_op`` but with a slot-based program-cache override.

        On program cache miss, builds the program and records which per-core /
        common runtime args and which CBs hold ``io_tensors`` buffer addresses.
        On cache hit, updates only those words and dynamic CB bindings — no
        memcpy of full per-core runtime-arg vectors (unlike ``generic_op``).

        Returns ``None``; use the preallocated output tensors passed in ``io_tensors``.
        )doc");
}

}  // namespace ttnn::operations::experimental::generic::detail
