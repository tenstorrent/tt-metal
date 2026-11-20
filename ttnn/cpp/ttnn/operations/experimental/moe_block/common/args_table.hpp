// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-core runtime values kept in device DRAM instead of runtime args (the args table).
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
//  2. An address arg is a source buffer's address plus an offset. The block holds only its offset, and lists the arg
//     after the args (a fixup: arg index, source). The sources' addresses go to the kernel's common args.
//  3. The kernel's per-core runtime args are removed (but its kept args, see below); it gets the table's place and the
//     sources as common args, and named compile-time args (kernels/args_table_names.hpp) saying where things sit.
// The programs are then identical across chips but for kept args (merge_chip_workload makes them one program for the
// mesh, the kept args that differ by chip per-device runtime args).
//
// Kept args: a [start, start + count) copy of a core's args stays as its runtime args (e.g. a fabric connection's,
// which the fabric API reads with get_arg_val); the block keeps them too, at the same indices, so no index moves.
//
// What the kernel does (kernels/moe_block_args.hpp)
// -------------------------------------------------
// moe_block_args::load() reads its core's block from DRAM into L1, adds each fixup's source address to its arg, and
// from then on moe_block_args::arg(i) returns what get_arg_val(i) returned before. Public dataflow APIs only.
// The block lands in a CB the kernel's cores have (ArgsTableKernel::block_cb, at l1_offset), or at an L1 address the
// common args hold (block_base + l1_offset).
//
// A program-cache hit (write_program_addresses) rewrites every source address and every block address in the common
// args, re-points every arena-backed CB, and rewrites any address arg left as a plain runtime arg. Nothing per core,
// and no check of what moved.

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

// One program per chip, the chip's coordinate with it.
using ChipPrograms = std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>;

struct ArgsTableFixup {
    uint32_t index = 0;   // the arg (in the core's args as built)
    uint32_t source = 0;  // the source (an index into `sources`)
};
struct ArgsTableKernel {
    size_t kernel = 0;  // the kernel's index in every chip's program
    // its block's L1 address: block_cb's write pointer + this, or (no block_cb) block_base + this (64 B aligned)
    uint32_t l1_offset = 0;
    uint32_t bytes = 0;  // the L1 reserved there (the block must fit)
    // per chip (in `chips` order): per core, its fixups and its kept runtime args (start, count)
    std::vector<std::map<std::pair<size_t, size_t>, std::vector<ArgsTableFixup>>> fixups;
    std::vector<std::map<std::pair<size_t, size_t>, std::pair<uint32_t, uint32_t>>> kept;
    // the block in this CB of the kernel's cores (e.g. on an arena that moves: the kernel reads the CB's address)
    std::optional<uint8_t> block_cb;
};
struct ArgsTable {
    std::shared_ptr<const void> tables;  // nullptr: not converted (`why` says why; the programs are unchanged)
    // per converted kernel (in `kernels` order), per chip: its common args' first index (the same on every chip whose
    // programs merge; a chip whose kernel has more common args - e.g. a longer report - gets its own)
    std::vector<std::vector<uint32_t>> common_index;
};
// Common args a converted kernel gets at its common index: [0] table, [1] block bytes, [2] banks, [3] grid width,
// [4] the block's L1 address (block_base + l1_offset; 0 with a block_cb), [5] S, then the S source addresses.
inline constexpr uint32_t kArgsTableBlockAddress = 4, kArgsTableSources = 6;

// Converts `kernels` of the per-chip programs `chips` (the same kernels and CBs in each): see above. `sources`: each
// source's address as the args were built (an address arg minus its source's address is its offset). Returns the
// tables (owned with the cached program), or nullptr with `why` set when a kernel cannot be converted (the programs
// are then unchanged but for the kernels getting the table's named compile-time args off: fill_args_table_named_args).
ArgsTable runtime_args_to_dram(
    tt::tt_metal::distributed::MeshDevice& mesh,
    ChipPrograms& chips,
    const std::vector<ArgsTableKernel>& kernels,
    std::span<const uint32_t> sources,
    uint32_t block_base,
    std::string& why);

// The named compile-time args of kernels/moe_block_args.hpp (names: kernels/args_table_names.hpp). The kernel reads
// every one of them, so the host gives every kernel that includes moe_block_args.hpp all of them: the converter sets
// the table's, fill_args_table_named_args adds what is still missing as off (0), the rest are for the kernels that
// share a block with a compute kernel or keep their common args in it (the MoE head).
//
// Sets the named compile-time arg `name` of `kernel`: replaces its value when the kernel has it, else adds it.
void set_named_compile_time_arg(tt::tt_metal::KernelDescriptor& kernel, const std::string& name, uint32_t value);
// Adds every named compile-time arg of the args table that `kernel` has no value for, with the value 0 (the table off
// for a kernel whose args stay runtime args, or not yet converted). Call it on every kernel that includes
// moe_block_args.hpp, after the calls below and before the program is built; runtime_args_to_dram sets the table's
// values over it.
void fill_args_table_named_args(tt::tt_metal::KernelDescriptor& kernel);
// The kernel's common args sit in its block from word `at` on (moe_block_args::common reads them there).
void set_args_table_common_in_block(tt::tt_metal::KernelDescriptor& kernel, uint32_t at);
// A data-movement kernel hands the compute kernel on its core `words` words of its block, from word `at`, as one page
// of CB `cb` (after its own load()).
void set_args_table_peer_page_sender(tt::tt_metal::KernelDescriptor& kernel, uint32_t cb, uint32_t at, uint32_t words);
// A compute kernel reads its args from the page in CB `cb` (its common args from word `common_at` on).
void set_args_table_compute_reads_page(tt::tt_metal::KernelDescriptor& kernel, uint32_t cb, uint32_t common_at);

// The bytes of a block of `words` args and `fixups` fixups, as runtime_args_to_dram lays it out.
uint32_t args_block_bytes(uint32_t words, uint32_t fixups);

// Every BufferBinding of kernel.kernel on every chip becomes a fixup (its arg, the one source slot holding its buffer;
// offset 0) and is cleared. TT_FATAL if a bound buffer is no source or two, or the kernel has common buffer bindings.
// Resizes kernel.fixups / kernel.kept to the chips.
void fixups_from_buffer_bindings(
    ChipPrograms& chips, ArgsTableKernel& kernel, std::span<tt::tt_metal::Buffer* const> sources);

// What a cache hit writes into one cached program (kept in its factory's shared variables).
struct ProgramAddresses {
    // a kernel with its args in the table: its common index, and (no block_cb) the block's L1 address
    // sources[block_source] + block_offset
    struct TableKernel {
        tt::tt_metal::KernelHandle kernel = 0;
        uint32_t common_index = 0;
        std::optional<uint32_t> block_source;
        uint32_t block_offset = 0;
    };
    // an address arg left as a plain runtime / common arg: sources[source] + offset
    struct AddressArg {
        tt::tt_metal::KernelHandle kernel = 0;
        bool common = false;  // a common runtime arg (core unused)
        tt::tt_metal::CoreCoord core;
        uint32_t index = 0;
        uint32_t source = 0;
        uint32_t offset = 0;
    };
    // a CB on the arena: its index in the descriptor (= in the program's circular_buffers()), its offset there
    struct ArenaCb {
        uint32_t index = 0;
        uint32_t offset = 0;
    };
    std::shared_ptr<const void> tables;  // the DRAM tables, kept as long as the cached program
    std::vector<TableKernel> table_kernels;
    std::vector<AddressArg> args;
    std::vector<ArenaCb> arena_cbs;
};

// The converted kernels of chip `chip`'s program (block_cb kernels get no block_source).
std::vector<ProgramAddresses::TableKernel> table_kernels(
    const ArgsTable& table,
    const std::vector<ArgsTableKernel>& kernels,
    size_t chip,
    std::optional<uint32_t> block_source,
    uint32_t block_base_offset);

// Every CB of desc backed by a buffer: TT_FATAL unless that buffer is `arena` (nullptr: no CB may have a buffer), and
// TT_FATAL if a kernel still has buffer bindings (nothing would rewrite them).
std::vector<ProgramAddresses::ArenaCb> arena_backed_cbs(
    const tt::tt_metal::ProgramDescriptor& desc, const tt::tt_metal::Buffer* arena);

// A cache hit (or the first launch of a program whose descriptor holds offsets): writes this call's source addresses
// and block addresses into every table kernel's common args, every AddressArg, and re-points every arena CB.
void write_program_addresses(
    tt::tt_metal::Program& program,
    const ProgramAddresses& addresses,
    std::span<const uint32_t> sources,
    const tt::tt_metal::Buffer* arena);

}  // namespace ttnn::operations::experimental::moe_block
