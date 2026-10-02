// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader kernel's runtime arguments, the counterpart to combine_fabric2d_reader_ct_args.hpp.
//
// ReaderRtArgManager owns the one thing the two sides have to agree on: the order. It emplaces the args on
// the host and reads them back on the kernel, and only it can build a ReaderRtArgs.
//
// The args are the DRAM base addresses, which cannot be compile-time: a buffer's address is assigned by the
// allocator, so it describes an allocation and not a program, and a cached program is re-dispatched against
// whatever buffers the caller hands it. The manager keeps the buffers rather than their addresses so it
// emplaces them as buffers, which is what gets each position rebound on every dispatch.

#include "combine_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#endif

namespace cmbf2d {

#ifdef KERNEL_BUILD
inline uint32_t get_rt_arg(uint32_t idx) { return get_arg_val<uint32_t>(idx); }
#endif

struct ReaderRtArgManager;

struct ReaderRtArgs {
    uint32_t dram_in;
    uint32_t dram_out;
    uint32_t dram_fwd;
    uint32_t dram_meta;
    uint32_t dram_counts;
    uint32_t dram_region;
    uint32_t dram_expert_offsets;

private:
    friend struct ReaderRtArgManager;

#ifdef KERNEL_BUILD
    ReaderRtArgs() :
        dram_in(get_rt_arg(0)),
        dram_out(get_rt_arg(1)),
        dram_fwd(get_rt_arg(2)),
        dram_meta(get_rt_arg(3)),
        dram_counts(get_rt_arg(4)),
        dram_region(get_rt_arg(5)),
        dram_expert_offsets(get_rt_arg(6)) {}
#else
    ReaderRtArgs() = default;
#endif
};

struct ReaderRtArgManager {
#ifndef KERNEL_BUILD
    explicit ReaderRtArgManager(const op::DramBuffers& dram) : dram_(dram) {}

    void setup_rt_args(tt::tt_metal::KernelDescriptor& kernel_desc, const tt::tt_metal::CoreCoord& core) const {
        kernel_desc.emplace_runtime_args(
            core, {dram_.in, dram_.out, dram_.fwd, dram_.meta, dram_.counts, dram_.region, dram_.expert_offsets});
    }

private:
    op::DramBuffers dram_;
#else
    static ReaderRtArgs get_rt_args() { return ReaderRtArgs(); }
#endif
};

}  // namespace cmbf2d
