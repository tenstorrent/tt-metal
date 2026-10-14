// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

// Where each union kernel's source lives, one per RISC-V role. The union binaries carry both
// halves' bodies; see device/kernels/hybrid_reader.cpp.
struct MergedKernelSources {
    std::string reader;
    std::string writer;
    std::string compute;
};

// The grid rendezvous the union kernels run between the passes, in NoC coordinates the merge
// cannot derive on its own (it has no device to map logical cores through).
//
// `shared_semaphore_count` is how many ids the master zeroes before releasing; the barrier's own
// id sits above them and must survive.
struct PassBarrierPlan {
    tt::tt_metal::CoreCoord master_logical;
    uint32_t master_noc_x = 0;
    uint32_t master_noc_y = 0;
    uint32_t rect_x_start = 0;
    uint32_t rect_y_start = 0;
    uint32_t rect_x_end = 0;
    uint32_t rect_y_end = 0;
    uint32_t num_receivers = 0;
    // Two per core: both data-movement kernels arrive, so a release implies every core's writer --
    // and therefore its compute -- finished pass A.
    uint32_t total_arrivals = 0;
};

// What the merge decided. Logged by the caller on a cache miss; the arena footprint is also
// enforced against the arena's real per-core size inside overlay_circular_buffers.
struct MergeReport {
    // Per-role: every kernel has its own fused argument block, so the base the unified half sits
    // at differs between reader, writer and compute. Each union kernel carries its own pair as
    // defines; these are the same numbers, surfaced.
    struct Bases {
        uint32_t ct = 0;
        uint32_t rt = 0;
    };
    Bases reader;
    Bases writer;
    Bases compute;

    uint32_t arena_bytes_per_core = 0;
    uint32_t semaphore_count = 0;
    uint32_t barrier_semaphore_id = 0;
};

// Folds two single-implementation descriptors that target the SAME cores into one program.
//
// A program holds at most one kernel per processor per core, so this is not a concatenation: the
// six kernels become three, and each half's compile-time and runtime argument lists are joined
// behind the bases the union binaries were compiled with.
//
// `run_fused_pass` says whether pass A executes at all. When it is false the fused half's circular
// buffers are dropped -- nothing will touch them -- and the union kernels are built without the
// define that calls pass A. Both bodies are compiled in either way, so the argument bases have the
// same shape in both modes.
//
// `l1_arena` is required only when pass A runs; it backs both halves' circular buffers. The two
// passes never run at once, so their CBs are laid out from the arena's base independently and the
// program's L1 cost is the larger half, not the sum -- which is what lets both halves keep the
// whole core grid.
tt::tt_metal::ProgramDescriptor merge_halves(
    tt::tt_metal::ProgramDescriptor fused,
    tt::tt_metal::ProgramDescriptor unified,
    const MergedKernelSources& sources,
    bool run_fused_pass,
    tt::tt_metal::Buffer* l1_arena,
    const PassBarrierPlan& barrier,
    MergeReport& report);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
