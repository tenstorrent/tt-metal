// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The reader kernel's runtime arguments, shared by combine_fabric2d and by the routed expert's
// overlapped fork. Included inside the owning op's namespace, which supplies `op::DramBuffers`.

#pragma once

#ifndef KERNEL_BUILD
#include <variant>
#include <vector>
#endif

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
#ifdef CMBF2D_OVERLAPPED
    uint32_t dram_expert_table;
#endif

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
        dram_expert_offsets(get_rt_arg(6))
#ifdef CMBF2D_OVERLAPPED
        ,
        dram_expert_table(get_rt_arg(7))
#endif
    {
    }
#else
    ReaderRtArgs() = default;
#endif
};

struct ReaderRtArgManager {
#ifndef KERNEL_BUILD
    explicit ReaderRtArgManager(const op::DramBuffers& dram) : dram_(dram) {}

    void setup_rt_args(tt::tt_metal::KernelDescriptor& kernel_desc, const tt::tt_metal::CoreCoord& core) const {
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> args{
            dram_.in, dram_.out, dram_.fwd, dram_.meta, dram_.counts, dram_.region, dram_.expert_offsets};
        // Appended only when there is one; the kernel reads it only in the overlapped build.
        if (dram_.expert_table != nullptr) {
            args.push_back(dram_.expert_table);
        }
        kernel_desc.emplace_runtime_args(core, args);
    }

private:
    op::DramBuffers dram_;
#else
    static ReaderRtArgs get_rt_args() { return ReaderRtArgs(); }
#endif
};
