// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// An untilizer core's circular buffers and compute kernel. The standalone op and the routed expert's overlapped
// build lay them out identically, so the layout a reader addresses rows by lives in one place.

#include <cstdint>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Where an untilizer core's circular buffers end when laid out from `base`. The framework places a program's
// circular buffers from the L1 allocator base in declaration order, DRAM-aligned, and the overlapped build places
// them at the same offsets in its arena, so this is also where hand-placed memory on the core may start.
uint32_t untilizer_cbs_end(uint32_t base, const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args);

// cb_out first, so it is the memory a reader names as the batch ring. `arena` null: static circular buffers at the
// allocator base; otherwise the same layout from the arena's start.
void add_untilizer_cbs(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const tt::tt_metal::CoreRangeSet& core,
    tt::tt_metal::Buffer* arena);

// A plain untilize for BFLOAT16 tokens; for fp8 tokens one that also quantizes each scale block.
tt::tt_metal::KernelDescriptor untilizer_compute_kernel(
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const tt::tt_metal::CoreRangeSet& core);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
