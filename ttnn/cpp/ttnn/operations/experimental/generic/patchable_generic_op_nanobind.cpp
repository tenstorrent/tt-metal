// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "patchable_generic_op_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "device/patchable_generic_op_device_operation.hpp"
#include "device/patchable_generic_op_helpers.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {

namespace {

/// Patch the descriptor, wrap in MeshProgramDescriptor, and dispatch.
void patchable_generic_op_with_address_refresh(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const AddressSlots& address_slots) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    auto& desc_copy = mesh_program_descriptor.mesh_programs.back().second;
    patch_stale_descriptor(desc_copy, io_tensors, address_slots);

    (void)ttnn::prim::patchable_generic_op(io_tensors, mesh_program_descriptor);
}

}  // namespace

void bind_patchable_generic_op(nb::module_& mod) {
    nb::class_<AddressSlots>(mod, "AddressSlots", R"doc(
        Opaque mapping of every position in a ProgramDescriptor that references
        an IO tensor address (CB buffer pointers, per-core runtime args, common
        runtime args).  Computed once at build time via ``compute_address_slots``,
        stored by the fusion build cache, and passed to ``patchable_generic_op``
        on each launch to refresh stale addresses.
    )doc");

    mod.def(
        "compute_address_slots",
        &compute_address_slots,
        nb::arg("program_descriptor"),
        nb::arg("io_tensors"),
        R"doc(
        Compute the full address-slot mapping for a ProgramDescriptor.

        Must be called while buffer pointers and runtime arg addresses are
        valid (at build time, before tensors are freed).  Uses the same
        address-matching logic as ``discover_address_slots`` in the program
        factory.  The returned ``AddressSlots`` should be passed to
        ``patchable_generic_op`` on each launch.
        )doc");

    mod.def(
        "patchable_generic_op",
        &patchable_generic_op_with_address_refresh,
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        nb::arg("address_slots"),
        R"doc(
        Dispatch like ``generic_op`` but with slot-based program-cache overrides
        and stale-address patching.

        Patches the descriptor copy (CB buffer pointers, runtime args) from
        current IO tensor addresses using the pre-computed ``address_slots``
        mapping, then dispatches.  Skips patching entirely when all addresses
        match the build-time snapshot (zero-cost hot path).

        On program cache hit, the C++ program factory further patches the
        cached Program via its own slot-based mechanism.  On cache miss,
        the patched descriptor is used to construct a new Program.
        )doc");

    // Backward-compatible overload without AddressSlots (used by non-fusion callers).
    mod.def(
        "patchable_generic_op",
        [](const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
            patchable_generic_op_with_address_refresh(io_tensors, program_descriptor, AddressSlots{});
        },
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        R"doc(
        Dispatch like ``generic_op`` but with slot-based program-cache overrides.

        Overload without address-slot patching — for callers that always provide
        a descriptor with valid addresses (no fusion build cache).
        )doc");
}

}  // namespace ttnn::operations::experimental::generic::detail
