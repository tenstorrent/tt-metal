// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One program for the whole mesh.
//
// An op on a mesh builds one program per chip. On a 32-chip Galaxy the host then does its per-program work (cache
// lookup, address updates, dispatch commands) 32 times on every call. The MoE block's ops keep most of what differs
// between chips (each core's runtime args) in per-chip DRAM tables (args_table.hpp); what stays a runtime arg (a fabric
// connection's: its eth channel differs by chip) becomes a per-device runtime arg of one program over the mesh
// (ttnn/per_device_program_merge.hpp). merge_chip_workload puts the chips in the mesh range's order for that merge.
#pragma once

#include <string>
#include <vector>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <ttnn/per_device_program_merge.hpp>

namespace ttnn::operations::bringup::moe_block {

struct ChipWorkload {
    // one program over the chips' rectangle, or (no merge) the chips' own programs unchanged
    tt::tt_metal::WorkloadDescriptor workload;
    // per program of `workload` (in its order): the runtime-arg words that differ by chip (empty: none)
    std::vector<std::vector<ttnn::PerDeviceRuntimeArgsField>> per_device_runtime_args;
};

// A workload of one program per chip (each a one-coordinate range, together one rectangle of the mesh) as one program
// over the rectangle, the runtime-arg words that differ by chip per-device runtime args. When the programs differ in
// anything else it comes back unchanged and `why` (if given) says why.
ChipWorkload merge_chip_workload(tt::tt_metal::WorkloadDescriptor w, const char* op, std::string* why = nullptr);

// Adds program `index` of `chips` to `workload` (built from its descriptor, or `program` when given) with its
// per-device runtime args.
void add_chip_program(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const ChipWorkload& chips,
    size_t index,
    tt::tt_metal::Program program);

}  // namespace ttnn::operations::bringup::moe_block
