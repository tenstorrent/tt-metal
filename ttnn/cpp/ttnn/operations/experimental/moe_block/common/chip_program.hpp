// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One program for the whole mesh: an op that builds one program a chip, programs that differ only in their argument
// values (and in which cores carry a few semaphores), runs as one program - the host's per-program work (cache-hit
// patching, dispatch commands) done once instead of once a chip.
//
// The merge keeps one chip's program (the base) and, for every other chip, where its argument values differ - as the
// difference to the base's value, added at launch (the base is what the host rewrites on a cache hit: an address that
// moves keeps the chips' differences): each chip's DRAM holds its own patch blocks (one MeshBuffer, the same address on
// every chip, each shard written with that chip's blocks when the program is built). A data-movement kernel with any
// per-chip value reads its core's block at its start (kernels/mb_chip.hpp, through mb_loop_main.hpp's kernel_main) and
// writes the values into its runtime args - the kernel's code reads its args as before. No chip id is needed on the
// device: the block is at the same place on every chip, its contents are the chip's own.
//
// What may differ between the chips' programs: runtime-arg and common-runtime-arg values (a compute kernel's through a
// data-movement kernel on its cores), their counts (padded), and the cores a semaphore sits on (the union). An arg the
// host rewrites from an address must take it from the same source on every chip (the op checks; a bound buffer's arg
// may not differ at all). Everything else - kernels, sources, compile-time args, defines, core ranges, CBs - must
// match: an op makes its programs alike (e.g. its port kernels on the union of every chip's port cores, idle where a
// chip has none).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

struct ChipProgram {
    tt::tt_metal::ProgramDescriptor desc;
    // the chips' patch tables (none when every chip's program is the same): owned with the cached program
    std::shared_ptr<const void> table;
};

// The chips' programs (each chip's coordinate and program) as one program, or nullopt with `why` set when they differ
// in more than argument values / semaphore cores (the caller keeps one program a chip).
std::optional<ChipProgram> merge_chip_programs(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>> chips,
    std::string& why);

// A workload of one program a chip (each a one-coordinate range, together one rectangle of the mesh) as one program
// over the rectangle - or unchanged (logged at debug, `op` naming it) when the programs cannot merge.
std::pair<tt::tt_metal::WorkloadDescriptor, std::shared_ptr<const void>> merge_chip_workload(
    tt::tt_metal::WorkloadDescriptor w, tt::tt_metal::distributed::MeshDevice& mesh, const char* op);

}  // namespace ttnn::operations::experimental::moe_block
