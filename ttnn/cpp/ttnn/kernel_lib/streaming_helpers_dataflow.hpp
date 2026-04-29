// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"

namespace dataflow_kernel_lib {

using ckernel::ReduceDim;

// =============================================================================
// stream_axis_blocks — tile-streaming primitive for axis-reduce shaped ops
//
// Streams tiles to `cb_in` for one slice of the *preserved* axis, walking the
// *reduce* axis in block-chunked order. One call = one pass over a contiguous
// range of preserved-axis indices. Higher-level patterns (multi-pass, per-row,
// multicore, auxiliary tensors) compose by varying the slice and the number
// of calls — the helper itself stays oblivious.
//
// L1 trade-off — this is NOT the canonical reduction shape
//   The canonical way to write a reduction is to hold the full working set in
//   CBs and reduce in one shot — simplest, maximally pipelined, but CB size
//   scales with the dimension being processed. Large Ht or Wt does not fit.
//
//   This helper exists for the case where L1 is the binding constraint. When
//   composed with a per-row outer loop (Example 2 below), every CB is sized
//   in terms of just two factors:
//     - BLOCK_SIZE along the reduce axis (capped, small — typically <= 8)
//     - the preserved-axis slice the caller passes per call (typically 1)
//   The actual input dimensions (Ht, Wt) do NOT enter the CB sizing. That
//   decouples L1 footprint from input shape — the same kernel handles
//   (1, 1, 32, 256) and (1, 1, 8192, 64000) without resizing CBs.
//
//   Cost: per-row init/uninit overhead, reduced inter-row pipelining, and
//   re-streaming the input N times for an N-pass algorithm. For shapes where
//   the canonical layout fits in L1, the canonical layout is usually
//   preferable. Reach for this primitive when L1 is the binding constraint.
//
// Vocabulary
//   reduce axis    : the axis being collapsed (W for REDUCE_ROW, H for REDUCE_COL)
//   preserved axis : the axis that survives in the output (H for REDUCE_ROW,
//                    W for REDUCE_COL)
//   block          : a contiguous chunk of BLOCK_SIZE tiles along the reduce
//                    axis. The reduce-axis extent must be divisible by
//                    BLOCK_SIZE; num_blocks is derived as extent / BLOCK_SIZE.
//
// Iteration order (REDUCE_ROW, collapse W):
//   for outer in [preserved_start, preserved_end):       // ht
//       for b in [0, num_blocks):                        // W-blocks
//           for inner in [0, BLOCK_SIZE):                // wt within block
//               tile_id = outer * Wt + b * BLOCK_SIZE + inner
//
// Iteration order (REDUCE_COL, collapse H):
//   for outer in [preserved_start, preserved_end):       // wt
//       for b in [0, num_blocks):                        // H-blocks
//           for inner in [0, BLOCK_SIZE):                // ht within block
//               tile_id = (b * BLOCK_SIZE + inner) * Wt + outer
//
// What this helper does NOT do
//   - scaler setup       → use prepare_reduce_scaler /
//                          prepare_partial_reduce_scalers
//   - multi-pass orchestration → caller's outer loop
//   - cross-core partial-sum gather → separate primitive (only needed when
//                          splitting the reduce axis across cores)
//   - padding suppression → handled at the FPU via a partial scaler tile;
//                          this helper streams every tile including the
//                          partially-valid last one.
//
// =============================================================================
// Example 1 — Whole-tensor single pass (simplest case)
//
// A row-wise sum reduction reader: stream the whole input once.
//
//   constexpr uint32_t cb_in = 0;
//   constexpr uint32_t cb_scaler = 2;
//
//   // Scaler: one tile of 1.0 for SUM (no /N).
//   dataflow_kernel_lib::prepare_reduce_scaler<
//       cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f);
//
//   const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);
//
//   dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_ROW>(
//       accessor, /*preserved_start=*/0, /*preserved_end=*/Ht,
//       Ht, Wt, BLOCK_SIZE);
//
// Compute pairs this with one call to compute_kernel_lib::accumulate_reduce
// using ReduceInputBlockShape::of(Ht, BLOCK_SIZE, NC).
// =============================================================================
// Example 2 — Per-row two-pass (H-safe variance / layernorm style)
//
// Each row is independent, so we serialize over rows. cb_mean / cb_variance
// only need to hold one tile at a time; L1 footprint is independent of Ht.
//
//   for (uint32_t outer = 0; outer < Ht; ++outer) {
//       // Pass 1: feeds compute's mean reduction.
//       dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_ROW>(
//           accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
//       // Pass 2: feeds compute's (x - mean)^2 reduction.
//       dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_ROW>(
//           accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
//   }
//
// Compute pairs this with an outer `for ht` loop using
// ReduceInputBlockShape::of(/*Ht=*/1, BLOCK_SIZE, /*NC=*/1).
// See ttnn/ttnn/operations/toy_variance/ for a working example.
// =============================================================================
// Example 3 — Multicore (split the preserved axis across cores)
//
// Rows are independent, so every core runs the full pipeline on its assigned
// row range — no cross-core sync, no semaphores. Only the descriptor changes
// vs. the per-row example: each core gets its own (start, end) RT args.
//
//   const uint32_t preserved_start = get_arg_val<uint32_t>(1);
//   const uint32_t preserved_end   = get_arg_val<uint32_t>(2);
//
//   for (uint32_t outer = preserved_start; outer < preserved_end; ++outer) {
//       dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_ROW>(
//           accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
//       dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_ROW>(
//           accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
//   }
//
// Each core's reader prepares its own scaler tile in its own CB; the partial
// scaler still applies on the last reduce-axis tile of every core (each core
// walks the full reduce extent).
// =============================================================================
// Example 4 — Auxiliary 1xW tensor (gamma/beta for layernorm pass 3)
//
// Auxiliary tensors that don't have an H dimension (γ, β) reuse the same
// helper with preserved range (0, 1) and a different accessor + CB. The
// tile-id formula collapses to `b * BLOCK_SIZE + inner` since outer = 0.
//
//   // Pass 3: stream x, γ, β alongside one another for one row at a time.
//   for (uint32_t outer = 0; outer < Ht; ++outer) {
//       // x : preserved range covers one ht
//       dataflow_kernel_lib::stream_axis_blocks<cb_x, ReduceDim::REDUCE_ROW>(
//           accessor_x, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
//       // γ : 1xW tensor, preserved range is (0, 1)
//       dataflow_kernel_lib::stream_axis_blocks<cb_gamma, ReduceDim::REDUCE_ROW>(
//           accessor_gamma, 0, 1, /*Ht=*/1, Wt, BLOCK_SIZE);
//       // β : same shape as γ
//       dataflow_kernel_lib::stream_axis_blocks<cb_beta, ReduceDim::REDUCE_ROW>(
//           accessor_beta, 0, 1, /*Ht=*/1, Wt, BLOCK_SIZE);
//   }
// =============================================================================
// Example 5 — H-reduction (REDUCE_COL, column-wise)
//
// Symmetric to the W-reduce case: blocks chunk H, preserved is W. The helper
// picks the column-major tile-id formula automatically.
//
//   dataflow_kernel_lib::prepare_reduce_scaler<
//       cb_scaler, PoolType::AVG, ReduceDim::REDUCE_COL>(1.0f / origin_H);
//
//   dataflow_kernel_lib::stream_axis_blocks<cb_in, ReduceDim::REDUCE_COL>(
//       accessor, /*preserved_start=*/0, /*preserved_end=*/Wt,
//       Ht, Wt, BLOCK_SIZE_H);
//
// For non-aligned H (origin_H % 32 != 0), pair with
// prepare_partial_reduce_scalers<..., ReduceDim::REDUCE_COL, partial_h>(...)
// — partial-axis correction is symmetric across REDUCE_ROW and REDUCE_COL.
// =============================================================================

/**
 * @brief Stream tiles for a slice of the preserved axis to `cb_in`, in
 *        reduce-axis-block order.
 *
 * Per outer index, walks num_blocks = (reduce-axis extent) / BLOCK_SIZE
 * blocks, each containing BLOCK_SIZE consecutive tiles along the reduce axis.
 * One tile is reserved, read, and pushed at a time (matches the streaming-
 * reduce helpers' per-tile-pop expectation on the compute side).
 *
 * @tparam cb_in       Circular buffer ID to push tiles to (must be constexpr)
 * @tparam reduce_dim  REDUCE_ROW (collapse W; preserved = H) or REDUCE_COL
 *                     (collapse H; preserved = W). REDUCE_SCALAR is rejected.
 * @tparam Accessor    TensorAccessor type (deduced from argument).
 *
 * @param accessor         Tile accessor for the source tensor.
 * @param preserved_start  Inclusive start index along the preserved axis.
 * @param preserved_end    Exclusive end index along the preserved axis. Must
 *                         be >= preserved_start; an empty range is a no-op.
 * @param Ht               Tile count along H. For REDUCE_COL this is the
 *                         reduce-axis extent and must be divisible by BLOCK_SIZE.
 * @param Wt               Tile count along W. For REDUCE_ROW this is the
 *                         reduce-axis extent and must be divisible by BLOCK_SIZE.
 *                         Used as the row stride in tile-id math for both axes.
 * @param BLOCK_SIZE       Block size in tiles along the reduce axis. Must
 *                         divide the reduce-axis extent.
 */
template <uint32_t cb_in, ReduceDim reduce_dim, typename Accessor>
FORCE_INLINE void stream_axis_blocks(
    const Accessor& accessor,
    uint32_t preserved_start,
    uint32_t preserved_end,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t BLOCK_SIZE);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/streaming_helpers_dataflow.inl"
