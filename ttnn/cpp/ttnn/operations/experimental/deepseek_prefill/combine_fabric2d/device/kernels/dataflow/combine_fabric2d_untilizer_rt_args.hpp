// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The untilizer kernel's runtime arguments, the counterpart to combine_fabric2d_untilizer_ct_args.hpp.
//
// UntilizerRtArgManager owns the one thing the two sides have to agree on: the order. It emplaces the args on
// the host and reads them back on the kernel, and only it can build an UntilizerRtArgs.
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

struct UntilizerRtArgManager;

struct UntilizerRtArgs {
    uint32_t dram_in;
    uint32_t dram_counts;
    uint32_t dram_region;
    uint32_t dram_expert_offsets;

private:
    friend struct UntilizerRtArgManager;

#ifdef KERNEL_BUILD
    UntilizerRtArgs() :
        dram_in(get_arg_val<uint32_t>(0)),
        dram_counts(get_arg_val<uint32_t>(1)),
        dram_region(get_arg_val<uint32_t>(2)),
        dram_expert_offsets(get_arg_val<uint32_t>(3)) {}
#else
    UntilizerRtArgs() = default;
#endif
};

struct UntilizerRtArgManager {
#ifndef KERNEL_BUILD
    explicit UntilizerRtArgManager(const op::DramBuffers& dram) : dram_(dram) {}

    void setup_rt_args(tt::tt_metal::KernelDescriptor& kernel_desc, const tt::tt_metal::CoreCoord& core) const {
        kernel_desc.emplace_runtime_args(core, {dram_.in, dram_.counts, dram_.region, dram_.expert_offsets});
    }

private:
    op::DramBuffers dram_;
#else
    static UntilizerRtArgs get_rt_args() { return UntilizerRtArgs(); }
#endif
};

}  // namespace cmbf2d
