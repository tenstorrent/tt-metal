// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One program for the whole mesh.
//
// An op on a mesh builds one program per chip. On a 32-chip Galaxy the host then does its per-program work (cache
// lookup, address updates, dispatch commands) 32 times on every call. The MoE block's ops keep what differs between
// chips (each core's runtime args) in per-chip DRAM tables (args_table.hpp), so their chips' programs come out the
// same. merge_chip_workload then runs them as one program over the mesh.
//
// The chips' programs may differ only in which cores a semaphore is on: the merged program has the semaphore on the
// union of those cores. If anything else differs, there is no merge: the op keeps one program per chip, and `why`
// says what differs.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::moe_block {

// A workload of one program per chip (each a one-coordinate range, together one rectangle of the mesh) as one program
// over the rectangle. When the programs differ it comes back unchanged and `why` (if given) says why.
tt::tt_metal::WorkloadDescriptor merge_chip_workload(
    tt::tt_metal::WorkloadDescriptor w, const char* op, std::string* why = nullptr);

}  // namespace ttnn::operations::experimental::moe_block
