// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader kernel's runtime arguments. One field list: the host constructor takes what the program factory
// already has and derives every field from it, the kernel constructor reads the same fields back in the same
// order. Same contract as combine_fabric2d_reader_ct_args.hpp, for the args that cannot be compile-time.
//
// Which is these: a buffer's address is assigned by the allocator, so it describes an allocation and not a
// program. Held in compile-time args it survived into every program cache hit, and a hit re-dispatched
// against buffers the caller had reallocated read its inputs and wrote its output at the previous call's
// addresses.
//
// Being runtime args is not by itself what fixes that. A uint32_t runtime arg is embedded when the program is
// built and never touched again -- same staleness, one arg list further along. What fixes it is the
// BufferBinding the framework patches on every dispatch, so this struct states both halves off the one field
// order: to_rt_arg_vector() is the values the args start at, buffer_bindings() is the positions the framework
// overwrites. Add a field and both follow; the kernel reads the same index either way.

#include "combine_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#endif

namespace cmbf2d {

#ifdef KERNEL_BUILD
// Reads like get_compile_time_arg_val on the other side of the pair, so the two constructors below stay
// visibly symmetric.
inline uint32_t get_rt_arg(uint32_t idx) { return get_arg_val<uint32_t>(idx); }
#endif

struct ReaderRtArgs {
    uint32_t dram_in;
    uint32_t dram_out;
    uint32_t dram_fwd;
    uint32_t dram_meta;
    uint32_t dram_counts;
    uint32_t dram_region;
    uint32_t dram_expert_offsets;

#ifndef KERNEL_BUILD
    explicit ReaderRtArgs(const op::DramBuffers& dram) :
        dram_in(static_cast<uint32_t>(dram.in->address())),
        dram_out(static_cast<uint32_t>(dram.out->address())),
        dram_fwd(static_cast<uint32_t>(dram.fwd->address())),
        dram_meta(static_cast<uint32_t>(dram.meta->address())),
        dram_counts(static_cast<uint32_t>(dram.counts->address())),
        dram_region(static_cast<uint32_t>(dram.region->address())),
        dram_expert_offsets(static_cast<uint32_t>(dram.expert_offsets->address())),
        dram_(dram) {}

    std::vector<uint32_t> to_rt_arg_vector() const {
        return {dram_in, dram_out, dram_fwd, dram_meta, dram_counts, dram_region, dram_expert_offsets};
    }

    // The same order as to_rt_arg_vector(), as arg positions the framework rewrites per dispatch. Without
    // these the addresses above would be baked into the cached program, which is the bug this file exists for.
    tt::tt_metal::KernelDescriptor::BufferBindings buffer_bindings(const tt::tt_metal::CoreCoord& core) const {
        tt::tt_metal::KernelDescriptor::BufferBindings bindings;
        uint32_t idx = 0;
        for (auto* buffer :
             {dram_.in, dram_.out, dram_.fwd, dram_.meta, dram_.counts, dram_.region, dram_.expert_offsets}) {
            bindings.push_back(tt::tt_metal::BufferBinding{core, idx++, buffer});
        }
        return bindings;
    }
#else
    ReaderRtArgs() :
        dram_in(get_rt_arg(0)),
        dram_out(get_rt_arg(1)),
        dram_fwd(get_rt_arg(2)),
        dram_meta(get_rt_arg(3)),
        dram_counts(get_rt_arg(4)),
        dram_region(get_rt_arg(5)),
        dram_expert_offsets(get_rt_arg(6)) {}
#endif

#ifndef KERNEL_BUILD
private:
    // Held only to state the bindings above; the args themselves are the uint32_t fields.
    const op::DramBuffers& dram_;
#endif
};

}  // namespace cmbf2d
