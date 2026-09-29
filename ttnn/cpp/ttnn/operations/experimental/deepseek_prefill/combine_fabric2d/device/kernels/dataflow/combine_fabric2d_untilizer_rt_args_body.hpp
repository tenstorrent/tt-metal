// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The untilizer kernel's runtime arguments, shared by combine_fabric2d and by the routed expert's
// overlapped fork. Included inside the owning op's namespace, which supplies `op::DramBuffers`.

#pragma once

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
#ifdef CMBF2D_OVERLAPPED
    uint32_t dram_expert_table;
#endif

private:
    friend struct UntilizerRtArgManager;

#ifdef KERNEL_BUILD
    UntilizerRtArgs() :
        dram_in(get_arg_val<uint32_t>(0)),
        dram_counts(get_arg_val<uint32_t>(1)),
        dram_region(get_arg_val<uint32_t>(2)),
        dram_expert_offsets(get_arg_val<uint32_t>(3))
#ifdef CMBF2D_OVERLAPPED
        ,
        dram_expert_table(get_arg_val<uint32_t>(4))
#endif
    {
    }
#else
    UntilizerRtArgs() = default;
#endif
};

struct UntilizerRtArgManager {
#ifndef KERNEL_BUILD
    explicit UntilizerRtArgManager(const op::DramBuffers& dram) : dram_(dram) {}

    void setup_rt_args(tt::tt_metal::KernelDescriptor& kernel_desc, const tt::tt_metal::CoreCoord& core) const {
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> args{
            dram_.in, dram_.counts, dram_.region, dram_.expert_offsets};
        // Appended only when there is one; the kernel reads it only in the overlapped build.
        if (dram_.expert_table != nullptr) {
            args.push_back(dram_.expert_table);
        }
        kernel_desc.emplace_runtime_args(core, args);
    }

private:
    op::DramBuffers dram_;
#else
    static UntilizerRtArgs get_rt_args() { return UntilizerRtArgs(); }
#endif
};
