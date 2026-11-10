// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-core runtime values kept in device DRAM instead of runtime args.
//
// Why this exists
// ---------------
// Every launch of a program writes its runtime args into each device's dispatch queue: for a program over a whole
// mesh, once per chip. On a Galaxy that copy is most of an op's host time (the experts + combine: 127 KB of args a
// chip, 32 chips, ~5 MB a call). Most of those values never change between calls: they only depend on the plan (which
// core talks to which, ring offsets, chip positions). This moves them into a table in each chip's DRAM, written once
// when the program is built; a launch then carries only the few values that do change between calls (buffer
// addresses, as common runtime args).
//
// It also makes the chips' programs identical: what differs between chips lives in each chip's own copy of the table,
// at the same address on every chip. So one program serves the whole mesh with no per-chip patching on the device.
//
// What the host does (runtime_args_to_dram)
// -----------------------------------------
// For each kernel the caller converts, given one program per chip (alike but for argument values):
//  1. Each core's runtime args (all of them, in their order) become that core's block in the kernel's table: a DRAM
//     MeshBuffer with one page per core of the grid (page = y * grid width + x), at the same address on every chip,
//     each chip's pages written with that chip's values.
//  2. The args bound to a buffer (BufferBinding) must sit at the same index with the same buffer on every core and
//     chip. Their values move to common runtime args (CommonBufferBinding), so a program-cache hit patches them as
//     usual; the kernel writes them back into its block's slots after loading it.
//  3. The kernel's per-core runtime args are removed; it gets the table's place as common args and two defines.
// The programs are then identical across chips (merge_chip_workload makes them one program for the mesh, with no
// per-chip patching).
//
// What the kernel does (kernels/moe_block_args.hpp)
// ------------------------------------------
// moe_block_args::load() reads its core's block from DRAM into an L1 scratch region the caller reserved (a CB the
// kernel's cores have, at a byte offset), patches in the bound buffer addresses, and from then on
// moe_block_args::arg(i) returns what get_arg_val(i) returned before. Public dataflow APIs only.
//
// Common args the converter appends (at MOE_BLOCK_ARGS_CA): the table's address, the block bytes, the DRAM banks, the
// grid width, the number of bound args n, then n pairs (block index, buffer address - the CommonBufferBinding).

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

// Where a converted kernel's block lands in L1: a CB its cores have, a byte offset in it (64 B aligned) and the bytes
// reserved there (the block must fit).
struct ArgsScratch {
    uint32_t cb = 0;
    uint32_t offset = 0;
    uint32_t bytes = 0;
    // args from this index on stay runtime args (at index 0 on): e.g. a fabric connection's, which the fabric API
    // reads with get_arg_val (build_from_args); they must then be the same on every chip for one program a mesh
    uint32_t keep_from = UINT32_MAX;
};

// Converts the kernels named by scratch (index = the kernel's index in each program; nullopt: left as it is) of the
// per-chip programs `chips` (the same kernels, cores, CBs in each): see above. Returns the tables (owned with the
// cached program), or nullptr with `why` set when a kernel cannot be converted (its programs are then unchanged).
std::shared_ptr<const void> runtime_args_to_dram(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>& chips,
    const std::vector<std::optional<ArgsScratch>>& scratch,
    std::string& why);

// ---- The fixup form (kernels/moe_block_args.hpp MOE_BLOCK_ARGS_FIXUP): for ops whose address args are a buffer's
// address plus a per-arg offset (their own patch lists, not BufferBindings). Each converted kernel's per-core args go
// to the table as built, but each address arg holds only its offset; the block lists those args (fixups: arg index,
// source), and the kernel adds the sources' current addresses (its common args) when it loads the block. A
// program-cache hit rewrites only the sources' addresses and the block's L1 address in the common args - nothing per
// core. Kept args: a [start, start + count) copy of a core's args stays as its runtime args (e.g. a fabric
// connection's, which the fabric API reads with get_arg_val) - the block keeps them too, at the same indices, so no
// index moves.
struct ArgsFix {
    uint32_t index = 0;  // the arg (in the core's args as built)
    uint32_t src = 0;    // the source (an index into `sources`)
};
struct ArgsFixupKernel {
    size_t kernel = 0;       // the kernel's index in every chip's program
    uint32_t l1_offset = 0;  // its block's L1 address: the scratch base + this (64 B aligned)
    uint32_t bytes = 0;      // the scratch there (the block must fit)
    // per chip (in `chips` order): per core, its fixups and its kept runtime args (start, count)
    std::vector<std::map<std::pair<size_t, size_t>, std::vector<ArgsFix>>> fixups;
    std::vector<std::map<std::pair<size_t, size_t>, std::pair<uint32_t, uint32_t>>> kept;
};
struct ArgsFixupResult {
    std::shared_ptr<const void> tables;  // nullptr: not converted (`why` says why; the programs are unchanged)
    // per converted kernel (in `kernels` order), per chip: its common args' first index (the same on every chip whose
    // programs merge; a chip whose kernel has more common args - e.g. a longer report - gets its own)
    std::vector<std::vector<uint32_t>> ca;
};
// Common args a converted kernel gets at ca: [0] table, [1] block bytes, [2] banks, [3] grid width, [4] the block's
// L1 address (scratch_base + l1_offset), [5] S, then the S source addresses.
inline constexpr uint32_t kArgsFixupL1 = 4, kArgsFixupSources = 6;
ArgsFixupResult runtime_args_to_dram_fixup(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>& chips,
    const std::vector<ArgsFixupKernel>& kernels,
    const std::vector<uint32_t>& sources,
    uint32_t scratch_base,
    std::string& why);

}  // namespace ttnn::operations::experimental::moe_block
