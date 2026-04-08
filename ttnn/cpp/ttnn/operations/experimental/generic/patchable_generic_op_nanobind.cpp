// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "patchable_generic_op_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "device/patchable_generic_op_device_operation.hpp"
#include "device/patchable_generic_op_helpers.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {

void bind_patchable_generic_op(nb::module_& mod) {
    mod.def(
        "patchable_generic_op",
        [](const std::vector<Tensor>& io_tensors,
           const tt::tt_metal::ProgramDescriptor& program_descriptor,
           const std::vector<std::pair<uint32_t, uint32_t>>& cb_io_tensor_map) {
            TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
            auto* mesh_device = io_tensors.front().device();
            TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

            tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
            mesh_program_descriptor.mesh_programs.emplace_back(
                ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

            // Refresh stale CBDescriptor.buffer pointers on the copy from current
            // IO tensors.  The fusion build cache may hold a ProgramDescriptor whose
            // buffer pointers have gone stale; cb_io_tensor_map (computed at build
            // time via compute_cb_io_tensor_map) records which CB maps to which IO
            // tensor so we can restore valid pointers before Program construction.
            auto& desc_copy = mesh_program_descriptor.mesh_programs.back().second;
            for (const auto& [cb_idx, io_idx] : cb_io_tensor_map) {
                TT_FATAL(
                    io_idx < io_tensors.size(),
                    "patchable_generic_op: cb_io_tensor_map io_idx {} out of range ({} io tensors)",
                    io_idx,
                    io_tensors.size());
                TT_FATAL(
                    cb_idx < desc_copy.cbs.size(),
                    "patchable_generic_op: cb_io_tensor_map cb_idx {} out of range ({} cbs)",
                    cb_idx,
                    desc_copy.cbs.size());
                desc_copy.cbs[cb_idx].buffer = io_tensors[io_idx].buffer();
            }

            (void)ttnn::prim::patchable_generic_op(io_tensors, mesh_program_descriptor);
        },
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        nb::arg("cb_io_tensor_map") = std::vector<std::pair<uint32_t, uint32_t>>{},
        R"doc(
        Dispatch like ``generic_op`` but with a slot-based program-cache override.

        On program cache miss, builds the program and records which per-core /
        common runtime args and which CBs hold ``io_tensors`` buffer addresses.
        On cache hit, updates only those words and dynamic CB bindings — no
        memcpy of full per-core runtime-arg vectors (unlike ``generic_op``).

        Args:
            io_tensors: Input and output tensors (output last).
            program_descriptor: The fused program descriptor.
            cb_io_tensor_map: Optional list of (cb_idx, io_tensor_idx) pairs.
                When provided, refreshes CBDescriptor.buffer pointers from the
                corresponding IO tensors before dispatch.  Used by the fusion
                build cache to keep stale buffer pointers valid.  Compute via
                ``compute_cb_io_tensor_map`` at build time.

        Returns ``None``; use the preallocated output tensors passed in ``io_tensors``.
        )doc");

    mod.def(
        "compute_cb_io_tensor_map",
        [](const tt::tt_metal::ProgramDescriptor& desc, const std::vector<Tensor>& io_tensors) {
            return compute_cb_io_tensor_map(desc, collect_io_tensor_addresses(io_tensors));
        },
        nb::arg("program_descriptor"),
        nb::arg("io_tensors"),
        R"doc(
        Compute the CB-to-IO-tensor index mapping for a ProgramDescriptor.

        For each CBDescriptor that has a buffer, matches its address against
        the IO tensor buffer addresses and returns a list of (cb_idx, io_tensor_idx)
        pairs.  Must be called while buffer pointers are still valid (at build time).

        This uses the same address-matching logic as ``discover_address_slots``
        in the program factory.  The returned map should be passed to
        ``patchable_generic_op`` on each launch to refresh stale buffer pointers.

        Args:
            program_descriptor: ProgramDescriptor with valid CB buffer pointers.
            io_tensors: The merged IO tensor list (inputs + outputs).

        Returns:
            List of (cb_idx, io_tensor_idx) pairs.
        )doc");
}

}  // namespace ttnn::operations::experimental::generic::detail
