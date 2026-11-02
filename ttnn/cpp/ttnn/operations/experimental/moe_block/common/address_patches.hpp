// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Address-only program-cache hits for a per-chip mesh workload. A program is built from a ProgramDescriptor whose
// address arguments are bound with emplace_runtime_args(Buffer*); take_address_patches records where each of them sits
// by source (an index into the op's list of buffers). On a hit, apply_address_patches compares each source's address
// with the one the program holds (its first argument tells) and rewrites that source's arguments only when it moved:
// a hit with nothing moved costs one runtime-arg lookup per source a chip, not a rewrite of every binding.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::moe_block {

struct AddressPatches {
    struct Arg {
        uint32_t kernel;  // the kernel's index in the descriptor (= its handle in the program built from it)
        bool common;      // a common runtime arg (core unused)
        tt::tt_metal::CoreCoord core;
        uint32_t index;
        uint16_t src;
    };
    std::vector<Arg> args;  // grouped by source
    // [begin, end) of each source's args: a hit reads one arg a source, never walks a source that did not move
    std::vector<std::pair<uint32_t, uint32_t>> groups;
};

// The address arguments of desc by source. sources[i]: source i's buffer (nullptr: absent). A buffer that is two
// sources is recorded as the first. Every binding must be to one of sources; desc may hold no tensor-backed CB.
AddressPatches take_address_patches(
    const tt::tt_metal::ProgramDescriptor& desc, std::span<tt::tt_metal::Buffer* const> sources);

// Rewrites the arguments of every source whose address moved (sources as at take_address_patches, this call's).
void apply_address_patches(
    tt::tt_metal::Program& program, const AddressPatches& patches, std::span<tt::tt_metal::Buffer* const> sources);

}  // namespace ttnn::operations::experimental::moe_block
