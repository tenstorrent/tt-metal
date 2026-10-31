// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::topk_large_indices::program {

// Shared host/device encoding for the compile-time row-reduction body.
enum class ComputeBodyMode : uint32_t {
    Classic = 0,
    FusedEndToEnd = 1,
    FusedSegmented = 2,
};

// The fused index stamp carries the chunk id in five bits, so a fused row or segment holds at most 32 chunks.
constexpr uint32_t max_fused_chunks = 32;

// Whether a row assigned to a core pair is actually halved: only a row of two or more chunks is. The host applies
// this to a length it knows (derive_core_row_assignments); under a metadata length it plans the pair from the
// physical width, so the kernels re-apply it to the real length. Otherwise a one-chunk row is still halved -- one
// half all -inf -- and the unfused merge that joins the halves orders tied values differently from the single
// fused reduction an eager call runs, so traced and eager indices disagree.
constexpr bool row_halved(uint32_t num_chunks) { return num_chunks >= 2; }

}  // namespace ttnn::operations::experimental::topk_large_indices::program
