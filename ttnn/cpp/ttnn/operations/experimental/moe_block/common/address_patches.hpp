// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fast program-cache hits: update only the buffer addresses that changed.
//
// Why this exists
// ---------------
// When an op is called again with the same shapes, its cached program is reused. The program's kernels hold buffer
// addresses in their runtime arguments (input, output, the block's arena, ...). Some of those buffers are allocated
// fresh on every call and may land at a different address. Before the program runs again, those arguments must be
// rewritten. Rewriting every address argument of every kernel on every chip on every call costs host time. Usually
// most buffers have not moved.
//
// How it works
// ------------
// When the program is built, take_address_patches records where each buffer's address appears: which kernel, which
// core, which argument index, grouped by buffer ("source"). It also records circular buffers placed inside a buffer
// (such as an op's region of the block's arena) and their offset.
// On a cache hit, apply_address_patches checks each source once: it reads one recorded argument and compares it with
// the buffer's current address. Only if the buffer moved does it rewrite that buffer's arguments and re-point its
// circular buffers. A call where nothing moved costs one lookup per buffer per chip.

#pragma once

#include <cstdint>
#include <span>
#include <utility>
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
    // A CB backed by a source's buffer (e.g. an op's region of the block's arena): the CB's index in the descriptor (=
    // in the program's circular_buffers()), its source, its offset in that buffer.
    struct Cb {
        uint32_t index;
        uint16_t src;
        uint32_t offset;
    };
    std::vector<Cb> cbs;
    // each source's address the CBs were last pointed at (a hit re-points a source's CBs only when it moved)
    mutable std::vector<uint64_t> cb_src_addr;
    // (later, first): at build source `later` was the same buffer as source `first`, so its args and CBs were recorded
    // as `first`'s. A hit must give both the same buffer again, else it fails (tt-metal's resolve_bindings rebuilds
    // instead): patching would give `later`'s args `first`'s address.
    std::vector<std::pair<uint16_t, uint16_t>> aliases;
};

// The address arguments of desc by source. sources[i]: source i's buffer (nullptr: absent). A buffer that is two
// sources is recorded as the first (AddressPatches::aliases). Every binding, and every tensor-backed CB's buffer, must
// be one of sources.
AddressPatches take_address_patches(
    const tt::tt_metal::ProgramDescriptor& desc, std::span<tt::tt_metal::Buffer* const> sources);

// Rewrites the arguments of every source whose address moved, and re-points its CBs (sources as at
// take_address_patches, this call's). TT_FATAL if two sources that were one buffer at build are not one now.
void apply_address_patches(
    tt::tt_metal::Program& program, const AddressPatches& patches, std::span<tt::tt_metal::Buffer* const> sources);

}  // namespace ttnn::operations::experimental::moe_block
