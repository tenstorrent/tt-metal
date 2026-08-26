// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt_stl/span.hpp>

// Nested per the sibling ports' convention: these names (`supported_by_codegen`, `is_demoted`) are
// generic enough that the bare data_movement namespace cannot hold one op's copy without colliding
// with the next.
namespace ttnn::operations::data_movement::permute_codegen {

// Correctness gate deciding whether a call lands on PermuteCodegenDeviceOperation vs. the native
// prim: row-major only, and declining the fused width-height permutation this port does not
// implement. Consulted by ttnn::permute's routing, by permute_force_codegen, and by
// prim::permute_codegen's validate, so all three agree on the supported scope.
// Never folds in perf demotions (see is_demoted below).
// `output_mem_config` is the caller's requested memory_config, not the input's: native permute
// accepts an interleaved input with a sharded output, which these kernels cannot produce.
bool supported_by_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config);

// Row-invariant CB batching. The reader reserves `kRmReadBatch` slots per round; the writer waits on
// the previous batch plus the current one before releasing either, so the deeper of those two holds
// is what the CB must be able to satisfy or the pair deadlocks.
inline constexpr uint32_t kRmReadBatch = 4;
inline constexpr uint32_t kRmWriteBatch = 4;
inline constexpr uint32_t kRmCbSlots = std::max(2 * kRmWriteBatch, kRmReadBatch);
static_assert(kRmCbSlots >= 2 * kRmWriteBatch, "writer_permute_rm_interleaved holds two write batches at once");
static_assert(kRmCbSlots >= kRmReadBatch, "reader_stick_interleaved_unified reserves a whole read batch at once");

// One row-invariant CB slot holds a whole row-major stick, so its byte size scales with the input's
// last dim. Shared by supported_by_codegen() (which rejects a stick too wide for kRmCbSlots of them
// to fit in one core's L1) and by the program factory (which sizes the CB from it); the two must
// agree or the gate would claim a config the factory cannot instantiate.
struct RmCbBudget {
    uint32_t slot_stride;  // bytes per CB slot; a pure function of the shape, dtype and alignments
    uint32_t max_slots;    // slots that fit under the caller-visible L1 frontier, read at call time
};
RmCbBudget rm_cb_budget(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config);

// True when the permutation leaves the last axis in place, which is what selects the row-invariant
// program factory (and with it the stick-sized CB above).
bool is_row_invariant(ttsl::Span<const uint32_t> dims);

// True when `dims` is a genuine permutation of [0, dims.size()): every axis in range, none repeated.
// Nothing upstream establishes this -- ttnn::permute normalizes each axis independently, so a
// repeated axis survives normalization -- and the kernels here derive their output extents and row
// counts from it, so both the scope gate and the prim check it.
bool is_permutation(ttsl::Span<const uint32_t> dims);

// Perf-only routing gate, consulted by ttnn::permute alone -- never by validate and never by
// permute_force_codegen. Demotes the blocked path as a whole -- any permutation that moves the last
// axis -- because it measures at parity with native; only the row-invariant path is routed here.
bool is_demoted(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims);

}  // namespace ttnn::operations::data_movement::permute_codegen
