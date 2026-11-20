// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "freeze_program_hashes_nanobind.hpp"

#include <nanobind/nanobind.h>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

// Defined in generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);

void bind_freeze_program_hashes(nb::module_& mod) {
    mod.def(
        "freeze_program_hashes",
        [](tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) {
            for (auto& entry : mesh_program_descriptor.mesh_programs) {
                auto& desc = entry.second;
                if (!desc.custom_program_hash) {
                    desc.custom_program_hash = compute_program_descriptor_hash(desc);
                }
            }
        },
        nb::arg("mesh_program_descriptor"),
        R"pbdoc(
            Store each program's structural hash in its custom_program_hash (where not set), in place: a
            descriptor reused across generic_op calls is then not re-hashed (kernel sources, compile-time args,
            core ranges) on every call. Only for a final descriptor - a later structural change is not seen.
        )pbdoc");
}

}  // namespace ttnn::operations::generic
