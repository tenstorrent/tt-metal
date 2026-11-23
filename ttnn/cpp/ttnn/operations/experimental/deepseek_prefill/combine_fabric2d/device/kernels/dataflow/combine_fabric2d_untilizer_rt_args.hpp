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

#ifndef KERNEL_BUILD
#include <variant>
#include <vector>
#endif

struct UntilizerRtArgManager;

struct UntilizerRtArgs {
    uint32_t dram_in;
    uint32_t dram_counts;
    uint32_t dram_region;
    uint32_t dram_expert_offsets;
    uint32_t dram_expert_table;  // zero when there is no table

private:
    friend struct UntilizerRtArgManager;

#ifdef KERNEL_BUILD
    // One kernel serves a whole untilizer group: the buffers are its common arguments, and each core's own
    // arguments are only what tells it apart (its index in the group, its produced-count semaphore).
    UntilizerRtArgs() :
        dram_in(get_common_arg_val<uint32_t>(0)),
        dram_counts(get_common_arg_val<uint32_t>(1)),
        dram_region(get_common_arg_val<uint32_t>(2)),
        dram_expert_offsets(get_common_arg_val<uint32_t>(3)),
        dram_expert_table(get_common_arg_val<uint32_t>(4)) {}
#else
    UntilizerRtArgs() = default;
#endif
};

struct UntilizerRtArgManager {
#ifndef KERNEL_BUILD
    explicit UntilizerRtArgManager(const op::DramBuffers& dram) : dram_(dram) {}

    // The buffers once, as common arguments of the kernel a group's cores share, and per core the position in
    // the group and its produced-count semaphore, which are all its cores differ by.
    void setup_rt_args(
        tt::tt_metal::KernelDescriptor& kernel_desc,
        const std::vector<tt::tt_metal::CoreCoord>& cores,
        const std::vector<uint32_t>& produced_slots) const {
        const auto expert_table = dram_.expert_table != nullptr
                                      ? std::variant<uint32_t, tt::tt_metal::Buffer*>(dram_.expert_table)
                                      : std::variant<uint32_t, tt::tt_metal::Buffer*>(0u);
        kernel_desc.emplace_common_runtime_args(
            {dram_.in, dram_.counts, dram_.region, dram_.expert_offsets, expert_table});
        for (uint32_t i = 0; i < cores.size(); i++) {
            kernel_desc.emplace_runtime_args(cores[i], {i, produced_slots[i]});
        }
    }

private:
    op::DramBuffers dram_;
#else
    static UntilizerRtArgs get_rt_args() { return UntilizerRtArgs(); }
#endif
};

}  // namespace cmbf2d
