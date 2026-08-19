// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ccl_helpers_schedule.hpp
 * @brief The COLLECTIVE SCHEDULE of a multi-kernel CCL op — one definition, shared by every
 *        kernel of the op and by its host program factory.
 *
 * The third member of the CCL helper family, alongside
 *   - @c ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp        (kernel-side fabric EGRESS)
 *   - @c ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp (its host companion)
 * and it is deliberately NEITHER of those: it holds no fabric state, allocates no packet header,
 * touches no CB, and issues no unpack/math/pack. It answers exactly one question, in one place:
 *
 *     "At this point in the collective, WHICH slice am I on, WHICH tiles does this chunk cover,
 *      and WHAT is this step supposed to do?"
 *
 * @par WHY THIS EXISTS — the footgun it removes.
 *   A reduction collective is built from three kernels that run concurrently on the same core set:
 *   a READER (pulls input + intermediate slices into CBs), a COMPUTE kernel (reduces them), and a
 *   WRITER (forwards or lands the result). All three must walk the SAME ring schedule in the SAME
 *   order with the SAME chunk boundaries. Before this header each one re-derived that schedule from
 *   scratch: @c reduce_scatter_minimal_async's ring reader and its ring reduction kernel carried
 *   BYTE-IDENTICAL copies of the 27-line step state machine, and the writer carried the same one
 *   with three extra flags. The even/odd chunk-splitting loop head was a third identical copy, and
 *   the per-slice tile-id walkers were duplicated between reader and writer.
 *
 *   Nothing enforced that agreement. It was maintained by copy-paste, and the failure modes are the
 *   worst two available: if the kernels disagree on a chunk boundary the op DEADLOCKS on a CB wait
 *   (the reader pushes a different tile count than the compute kernel waits for), and if they
 *   disagree on a step flag it SILENTLY CORRUPTS the output (a slice gets reduced twice, or not at
 *   all). Both survive review easily, because each kernel is individually self-consistent.
 *
 *   So the schedule gets the same treatment the fabric egress got in ccl_helpers_dataflow.hpp:
 *   ONE definition, consumed by every party, with drift made unexpressible rather than documented.
 *
 * @par WHY shared_with_host (and what that costs).
 *   The host program factory derives its own view of the same schedule — per-worker tile ranges,
 *   semaphore wait targets, chunks-per-sync — so host/kernel drift is the same bug class one level
 *   up. Living in @c shared_with_host/ means the factory can consume these functions directly, and
 *   means the schedule is reachable from host gtests: the step-flag table and the chunk walk can be
 *   swept exhaustively for every ring size and direction on the host, with no device and no fabric.
 *   That is the whole reason this is not a kernel-only header.
 *
 *   The cost is a hard discipline, inherited from @c hetergeneous_data_structs.hpp:
 *     - NO kernel-only headers, macros or intrinsics. In particular NO @c FORCE_INLINE (a
 *       dataflow_api macro), NO @c ASSERT, NO @c DPRINT, NO @c tt_l1_ptr. Everything here is plain
 *       C++17 over @c <cstdint> + @c <algorithm>. The members are small and header-inline, so the
 *       kernel compiler inlines them at -O3 without the macro.
 *     - NO types whose size differs across host and device (fixed-width integers and @c bool only).
 *     - Loud failure stays with the CALLER: instead of an in-header @c ASSERT this exposes
 *       predicates (@c is_supported_scatter_dim) that a kernel turns into a @c static_assert and the
 *       host turns into a @c TT_FATAL. Neither behaviour is baked in here.
 *
 * @par WHAT IS MODELLED.
 *   S1-S5: the schedule behind @c reduce_scatter_minimal_async's bidirectional RING variant,
 *   which @c all_reduce_async's reduce half also follows. S6-S7: the pieces its LINE variant
 *   shares between reader and writer (the line's slice cursor, the chunks-per-sync wait/signal
 *   cadence, the final-reduction mode split, and the plain channel/chunk walk). S8: the DIM-ZERO
 *   ring family's interleaved own/other chunk walk and its neighbour-first slice seed (the
 *   dim-zero LINE family composes S6-S7 with slice_B in the channel role).
 *
 * @par The RING schedule (S1-S5). Four nested levels, in this exact order,
 *   because that nesting is itself part of the contract the three kernels share:
 *
 *       batch  b  in [0, batches)              // input_tensor_B
 *         step i  in [0, ring_size/2 + 1)      // the ring walk; flags come from ring_rs_step_flags
 *           chan c  in [0, channels)           // slice_C; resets the chunk walk + bumps walker bases
 *             chunk    while tiles_read < total, alternating even/odd
 *
 *   The two halves of the ring are driven by ONE worker pair: a `direction` worker takes the even
 *   chunks and its opposite takes the odd ones, which is why every chunk carries an even/odd tag and
 *   an active/skip decision rather than simply being iterated.
 *
 * @par OWNERSHIP SPLIT (the same discipline as the dataflow helper).
 *   Owned here: loop control and termination, the step flag table, slice index + wraparound, chunk
 *   boundaries and the even/odd active/skip decision, and the per-slice tile-id walkers including
 *   their SKIP-PATH fast-forward (the walkers advance by the same tile count whether a chunk is
 *   processed or skipped, so keeping the fast-forward here is what makes a skipped chunk incapable
 *   of desynchronising them).
 *
 *   NOT owned here, and deliberately: every CB call, every semaphore wait/reset, every NoC or fabric
 *   issue, the reduction arithmetic, address generation (TensorAccessor is consumed by the caller,
 *   never wrapped), packet coalescing, and the chunks-per-sync semaphore protocol. This header tells
 *   a kernel WHAT to do next; it never does it. A schedule object is pure arithmetic over integers,
 *   which is exactly why it can be unit-tested on the host.
 *
 * @par USAGE — identical driver in all three kernels.
 * @code
 *   ttnn::ccl::schedule::RingRsSchedule sched(
 *       ring_size, input_tensor_B, slice_C, tile_granularity,
 *       start_tiles_read, start_tiles_to_read, direction);
 *
 *   while (sched.next_batch()) {
 *       // PER-BATCH: the ring walk restarts from the same first slice every batch, so the cursor is
 *       // constructed here rather than hoisted out — see RingSliceCursor's warning.
 *       ttnn::ccl::schedule::RingSliceCursor slice_cursor(my_chip_id, ring_size, direction);
 *       while (sched.next_step()) {
 *           const uint32_t slice_idx = slice_cursor.wrap();
 *           const auto& f = sched.flags();
 *           while (sched.next_channel()) {
 *               while (sched.next_chunk()) {
 *                   if (sched.skip()) { continue; }          // walkers already fast-forwarded
 *                   const uint32_t n = sched.tiles_this_chunk();
 *                   // reader : if (sched.reduce_interm()) { ... }   walker.next() x n
 *                   // compute: if (sched.reduce_interm()) { reduce n tiles }
 *                   // writer : if (f.write_to_remote())   { ... }   walker.next() x n
 *               }
 *           }
 *           slice_cursor.advance();
 *       }
 *       // batch epilogue (semaphore reset / cross-device barrier) stays caller-owned
 *   }
 * @endcode
 */

#include <algorithm>
#include <cstdint>

namespace ttnn::ccl::schedule {

// ===========================================================================
// S1 — the ring reduce-scatter STEP FLAG TABLE
// ===========================================================================

/**
 * @brief What one ring step is supposed to do. A SUPERSET across the three kernels: each reads only
 *        the fields it needs (the reader/compute use the reduce_* set, the writer the write_* set),
 *        which is precisely what lets one table serve all of them and guarantees they agree.
 *
 * All-false is the correct default for a step that does nothing, so every field is initialised.
 */
struct RingRsStepFlags {
    /// Process the even chunks this step (the half of each slice this worker's direction owns).
    bool even_chunks = false;
    /// Process the odd chunks this step (the half the opposite-direction worker owns).
    bool odd_chunks = false;
    /// An even chunk this step reduces (input + intermediate) rather than passing input through.
    bool reduce_even_chunks = false;
    /// An odd chunk this step reduces rather than passing through.
    bool reduce_odd_chunks = false;
    /// Final step: reduce THREE tensors (input + intermediate + output) instead of two.
    bool reduce_output = false;
    /// Writer: send over the fabric to the next chip, rather than landing locally.
    bool write_to_remote = false;
    /// Writer: destination is the intermediate tensor rather than the output tensor.
    bool write_to_interm = false;
    /// Writer: second-to-last step — even and odd chunks announce to DIFFERENT worker cores, so the
    /// chunks-per-sync counters must be kept separately per parity.
    bool separate_even_odd_sems = false;
};

/// Number of ring steps a bidirectional ring reduce-scatter walks (each worker covers half the ring
/// plus the terminal local step).
constexpr uint32_t ring_rs_num_steps(uint32_t ring_size) { return ring_size / 2 + 1; }

/**
 * @brief The step flag table — the single definition of the state machine that reader, compute and
 *        writer each used to carry their own copy of.
 *
 * `constexpr` and free of side effects so the host can sweep it exhaustively.
 *
 * @param step       Ring step index, in [0, ring_rs_num_steps(ring_size)).
 * @param ring_size  Devices in the ring. Even by construction for this schedule.
 * @param direction  This worker's ring direction; selects which chunk parity it owns.
 *
 * @note The `step == half - 1` case can COINCIDE with `step == 1` (it does at ring_size == 4). The
 *   ternaries are ordered so both conditions apply on that step, matching the pre-existing writer,
 *   whose combined `i == 1 || i == half - 1` branch had exactly this behaviour. `half - 1` is only
 *   evaluated when the step is neither 0 nor `half`, which cannot happen at ring_size == 2, so the
 *   subtraction never underflows.
 */
constexpr RingRsStepFlags ring_rs_step_flags(uint32_t step, uint32_t ring_size, bool direction) {
    const uint32_t half = ring_size / 2;
    RingRsStepFlags f{};
    if (step == 0) {
        // First step reads the local input slice only — nothing to reduce against yet.
        f.even_chunks = direction;
        f.odd_chunks = !direction;
        f.write_to_remote = true;
        f.write_to_interm = true;
    } else if (step == half) {
        // Terminal step: this worker's own slice, fully reduced, lands in the local output tensor.
        f.even_chunks = direction;
        f.odd_chunks = !direction;
        f.reduce_even_chunks = f.even_chunks;
        f.reduce_odd_chunks = f.odd_chunks;
        f.reduce_output = true;
    } else {
        f.even_chunks = true;
        f.odd_chunks = true;
        // Step 1 is the handover step: only the parity this worker owns has an intermediate to
        // reduce against yet. From step 2 on, both parities do.
        f.reduce_even_chunks = (step == 1) ? direction : true;
        f.reduce_odd_chunks = (step == 1) ? !direction : true;
        f.write_to_remote = true;
        f.write_to_interm = (step == half - 1) ? direction : true;
        f.separate_even_odd_sems = (step == half - 1);
    }
    return f;
}

// ===========================================================================
// S2 — the schedule driver (step / channel / chunk loop control)
// ===========================================================================

/**
 * @brief Drives the batch -> step -> channel -> chunk walk of a bidirectional ring reduce-scatter,
 *        and reports what each chunk is.
 *
 * Constructed from arguments EVERY one of the three kernels already has, so adopting it needs no new
 * compile-time or runtime args on any of them and no host-side change. (The slice index and the
 * tile-id walkers are deliberately separate types below, because the compute kernel needs neither
 * and should not be made to carry their arguments.)
 *
 * The four `next_*()` calls are a driver, not iterators: each advances its level, resets the levels
 * below it, and returns false when that level is exhausted. Nest them exactly as shown in the file
 * banner.
 */
class RingRsSchedule {
public:
    /**
     * @param ring_size            Devices in the ring (even).
     * @param batches              Outer batch count (input_tensor_B).
     * @param channels             Per-step channel count (slice_C).
     * @param tile_granularity     Maximum tiles per chunk. Must be <= the DEST register capacity,
     *                             since a chunk is what the compute kernel accumulates in one
     *                             tile_regs acquire/commit pass — assert that at the call site
     *                             against @c compute_kernel_lib::DEST_AUTO_LIMIT.
     * @param start_tiles_read     This worker's first tile index within a channel.
     * @param start_tiles_to_read  This worker's end tile index within a channel (exclusive).
     * @param direction            Ring direction; selects the chunk parity this worker owns.
     */
    RingRsSchedule(
        uint32_t ring_size,
        uint32_t batches,
        uint32_t channels,
        uint32_t tile_granularity,
        uint32_t start_tiles_read,
        uint32_t start_tiles_to_read,
        bool direction) :
        ring_size_(ring_size),
        num_steps_(ring_rs_num_steps(ring_size)),
        batches_(batches),
        channels_(channels),
        granularity_(tile_granularity),
        start_tiles_read_(start_tiles_read),
        total_tiles_(start_tiles_to_read),
        direction_(direction) {}

    // --- level 1: batch -------------------------------------------------------------
    /// Advance to the next batch, resetting the step walk. False when batches are exhausted.
    bool next_batch() {
        if (batch_started_) {
            ++batch_;
        } else {
            batch_started_ = true;
        }
        if (batch_ >= batches_) {
            return false;
        }
        step_started_ = false;
        step_ = 0;
        return true;
    }

    // --- level 2: ring step ---------------------------------------------------------
    /// Advance to the next ring step, refreshing flags() and resetting the channel walk.
    bool next_step() {
        if (step_started_) {
            ++step_;
        } else {
            step_started_ = true;
        }
        if (step_ >= num_steps_) {
            return false;
        }
        flags_ = ring_rs_step_flags(step_, ring_size_, direction_);
        channel_started_ = false;
        channel_ = 0;
        return true;
    }

    // --- level 3: channel -----------------------------------------------------------
    /// Advance to the next channel, resetting the chunk walk to this worker's tile range.
    bool next_channel() {
        if (channel_started_) {
            ++channel_;
        } else {
            channel_started_ = true;
        }
        if (channel_ >= channels_) {
            return false;
        }
        chunk_started_ = false;
        tiles_read_ = start_tiles_read_;
        tiles_this_chunk_ = 0;
        is_even_chunk_ = true;
        return true;
    }

    // --- level 4: chunk -------------------------------------------------------------
    /**
     * @brief Advance to the next chunk of this channel. False when the channel's tile range is
     *        exhausted.
     *
     * Chunk parity alternates every call and the tile count is halved on even chunks, so the two
     * directions interleave over one slice. A chunk of ZERO tiles is legal and is reported as
     * skipped: it occurs when an even chunk sees a single tile remaining (`remaining / 2 == 0`). The
     * parity still flips, so the following odd chunk takes that tile and the walk always progresses.
     */
    bool next_chunk() {
        if (chunk_started_) {
            tiles_read_ += tiles_this_chunk_;
            is_even_chunk_ = !is_even_chunk_;
        } else {
            chunk_started_ = true;
        }
        if (tiles_read_ >= total_tiles_) {
            return false;
        }
        const uint32_t remaining = total_tiles_ - tiles_read_;
        tiles_this_chunk_ = is_even_chunk_ ? std::min(remaining / 2, granularity_) : std::min(remaining, granularity_);
        return true;
    }

    // --- what this position IS ------------------------------------------------------
    /// Flags for the current ring step.
    const RingRsStepFlags& flags() const { return flags_; }
    uint32_t batch_idx() const { return batch_; }
    uint32_t step_idx() const { return step_; }
    uint32_t channel_idx() const { return channel_; }
    /// Tiles covered by the current chunk (0 .. tile_granularity).
    uint32_t tiles_this_chunk() const { return tiles_this_chunk_; }
    bool is_even_chunk() const { return is_even_chunk_; }

    /**
     * @brief This chunk carries no work for this worker: it belongs to the opposite direction's
     *        parity this step, or it is the zero-tile chunk.
     *
     * Callers holding tile-id walkers must still keep them moving across a skipped chunk — call
     * @c SliceRowWalker::advance(tiles_this_chunk()) / @c SequentialTileWalker::advance(...) before
     * continuing, exactly as the pre-existing reader and writer did in their skip branch.
     */
    bool skip() const {
        return (is_even_chunk_ && !flags_.even_chunks) || (!is_even_chunk_ && !flags_.odd_chunks) ||
               tiles_this_chunk_ == 0;
    }

    /// This chunk reduces against the intermediate tensor rather than passing input straight
    /// through — the fold of chunk parity against this step's reduce_even/odd flags.
    bool reduce_interm() const {
        return (is_even_chunk_ && flags_.reduce_even_chunks) || (!is_even_chunk_ && flags_.reduce_odd_chunks);
    }

private:
    // Immutable configuration.
    uint32_t ring_size_ = 0;
    uint32_t num_steps_ = 0;
    uint32_t batches_ = 0;
    uint32_t channels_ = 0;
    uint32_t granularity_ = 0;
    uint32_t start_tiles_read_ = 0;
    uint32_t total_tiles_ = 0;
    bool direction_ = false;

    // Walk state. The *_started_ flags exist so the first next_*() call sets up index 0 rather than
    // advancing past it, which keeps every level a plain `while (next_x())` at the call site.
    uint32_t batch_ = 0;
    uint32_t step_ = 0;
    uint32_t channel_ = 0;
    uint32_t tiles_read_ = 0;
    uint32_t tiles_this_chunk_ = 0;
    bool batch_started_ = false;
    bool step_started_ = false;
    bool channel_started_ = false;
    bool chunk_started_ = false;
    bool is_even_chunk_ = true;
    RingRsStepFlags flags_{};
};

// ===========================================================================
// S3 — slice index walk
// ===========================================================================

/// First slice a ring reduce-scatter worker processes: the one belonging to the device half-way
/// across the ring. May exceed ring_size; RingSliceCursor wraps it on the first step.
constexpr uint32_t ring_rs_first_slice(uint32_t my_chip_id, uint32_t ring_size) { return my_chip_id + ring_size / 2; }

/**
 * @brief Walks the slice index one step per ring step, wrapping into [0, ring_size).
 *
 * Reader and writer both need this and the compute kernel does not, so it is a separate type rather
 * than state on RingRsSchedule — the compute kernel would otherwise have to pass a my_chip_id it
 * does not receive.
 *
 * A single conditional add/subtract is sufficient (rather than a modulo) because the starting index
 * is below 2 * ring_size and every subsequent step moves by one from an already-wrapped value. That
 * is the pre-existing kernels' arithmetic, preserved exactly.
 *
 * @warning THE CURSOR IS PER-BATCH. Every batch restarts the ring walk from the same first slice, so
 *   construct this INSIDE the `while (next_batch())` body (or call reset()). Hoisting it out of the
 *   batch loop leaves batch N+1 continuing where batch N stopped, which reads and writes the wrong
 *   slice on every step of every batch after the first — silently, with no hang. That is a bug this
 *   header's own host equivalence sweep caught, which is the reason for both this warning and the
 *   construction site shown in the file banner.
 */
class RingSliceCursor {
public:
    RingSliceCursor(uint32_t my_chip_id, uint32_t ring_size, bool direction) :
        slice_idx_(static_cast<int32_t>(ring_rs_first_slice(my_chip_id, ring_size))),
        first_slice_(static_cast<int32_t>(ring_rs_first_slice(my_chip_id, ring_size))),
        ring_size_(static_cast<int32_t>(ring_size)),
        direction_(direction) {}

    /// Build a cursor starting at an EXPLICIT (possibly out-of-range; wrapped on first use) slice.
    /// The dim-zero ring walk starts at my_chip_id -/+ 1 (dim_zero_ring_first_slice) instead of
    /// half-way across the ring; its advance and wrap are otherwise identical.
    static RingSliceCursor starting_at(int32_t first_slice, uint32_t ring_size, bool direction) {
        RingSliceCursor c(0, ring_size, direction);
        c.slice_idx_ = first_slice;
        c.first_slice_ = first_slice;
        return c;
    }

    /// Restart the walk at the first slice. Equivalent to re-constructing; use whichever makes the
    /// per-batch cadence more obvious at the call site.
    void reset() { slice_idx_ = first_slice_; }

    /// Wrap and return the slice index for the current step. Call once per step, before use.
    uint32_t wrap() {
        if (slice_idx_ < 0) {
            slice_idx_ += ring_size_;
        } else if (slice_idx_ >= ring_size_) {
            slice_idx_ -= ring_size_;
        }
        return static_cast<uint32_t>(slice_idx_);
    }

    /// Step to the next slice. Call once per step, after the step body.
    void advance() { slice_idx_ += direction_ ? -1 : 1; }

private:
    int32_t slice_idx_ = 0;
    int32_t first_slice_ = 0;
    int32_t ring_size_ = 0;
    bool direction_ = false;
};

// ===========================================================================
// S4 — slice tile-index origin
// ===========================================================================

/// Scatter dims this schedule covers. Dim 0 is scattered on batch and has its own (`dim_zero_*`)
/// kernel family and schedule, so it is excluded here. Callers turn this into a static_assert
/// (kernel) or TT_FATAL (host) — this header deliberately does not carry an assert of its own.
constexpr bool is_supported_scatter_dim(uint32_t dim) { return dim == 1 || dim == 2 || dim == 3; }

/**
 * @brief First tile index of slice @c slice_idx, for a tensor scattered on @c dim.
 * @return 0 for an unsupported dim; gate on @c is_supported_scatter_dim at the call site.
 */
constexpr uint32_t slice_tile_offset(
    uint32_t dim, uint32_t slice_idx, uint32_t slice_C, uint32_t slice_Ht, uint32_t slice_Wt) {
    if (dim == 3) {
        return slice_idx * slice_Wt;
    }
    if (dim == 2) {
        return slice_idx * slice_Ht * slice_Wt;
    }
    if (dim == 1) {
        return slice_idx * slice_C * slice_Ht * slice_Wt;
    }
    return 0;
}

// ===========================================================================
// S5 — tile-id walkers
// ===========================================================================

/**
 * @brief Row-major walk over a slice narrower than the full tensor row: emit @c slice_Wt
 *        consecutive tile ids, then jump by the full tensor width to the next row of the slice.
 *
 * Replaces the `get_next_input_tile_id` / `get_next_interm_tile_id` lambdas that reader and writer
 * each defined with identical bodies.
 */
class SliceRowWalker {
public:
    SliceRowWalker() = default;
    /**
     * @param slice_Wt   Tiles per row WITHIN the slice.
     * @param tensor_Wt  Tiles per row of the FULL tensor (the row stride to jump by).
     */
    SliceRowWalker(uint32_t slice_Wt, uint32_t tensor_Wt) : slice_Wt_(slice_Wt), tensor_Wt_(tensor_Wt) {}

    /// Set the slice origin. Called once per RING STEP, from slice_tile_offset() for the step's
    /// slice (plus any batch offset).
    void set_base(uint32_t base) { base_ = base; }

    /// Move the origin on to the next channel. Called at the END of each channel — the base
    /// ACCUMULATES across channels while the within-slice offsets below are re-seeded.
    void bump_base(uint32_t delta) { base_ += delta; }

    /// Re-seed the within-slice position at the START of each channel, to this worker's start
    /// offsets. Deliberately separate from set_base(): the two are reset on different cadences.
    void reset_offsets(uint32_t pages_read_in_row, uint32_t row_offset) {
        pages_read_in_row_ = pages_read_in_row;
        row_offset_ = row_offset;
    }

    /// Next tile id.
    uint32_t next() {
        const uint32_t tile_id = base_ + row_offset_ + pages_read_in_row_;
        ++pages_read_in_row_;
        if (pages_read_in_row_ == slice_Wt_) {
            row_offset_ += tensor_Wt_;
            pages_read_in_row_ -= slice_Wt_;
        }
        return tile_id;
    }

    /// Fast-forward across a skipped chunk. Stepped one tile at a time so the row-wrap behaviour is
    /// identical to n calls to next() by construction; n is bounded by tile_granularity.
    void advance(uint32_t n) {
        for (uint32_t k = 0; k < n; ++k) {
            next();
        }
    }

private:
    uint32_t slice_Wt_ = 0;
    uint32_t tensor_Wt_ = 0;
    uint32_t base_ = 0;
    uint32_t pages_read_in_row_ = 0;
    uint32_t row_offset_ = 0;
};

/// Contiguous tile-id walk — the `get_next_output_tile_id` lambda, whose output slice is dense.
class SequentialTileWalker {
public:
    SequentialTileWalker() = default;

    /// Set the origin. Called once per RING STEP.
    void set_base(uint32_t base) { base_ = base; }

    /// Move the origin on to the next channel (accumulates, like SliceRowWalker::bump_base).
    void bump_base(uint32_t delta) { base_ += delta; }

    /// Re-seed the within-slice position at the START of each channel.
    void reset_offsets(uint32_t start) { offset_ = start; }

    uint32_t next() { return base_ + (offset_++); }

    void advance(uint32_t n) { offset_ += n; }

private:
    uint32_t base_ = 0;
    uint32_t offset_ = 0;
};

// ===========================================================================
// S6 — the chunks-per-sync cadence (line + dim-zero families)
// ===========================================================================

/**
 * @brief BOTH sides of the chunks-per-sync pairing: the reader's semaphore WAITS and the writer's
 *        semaphore SIGNALS, driven by one counter definition.
 *
 * The line and dim-zero reduce-scatter kernels throttle their cross-device handshake to one
 * semaphore exchange per @c chunks_per_sync chunks. Before this type, the reader spelled that as
 * "wait when count % cps == 0, then count++" and the writer as "count++, then signal when
 * count % cps == 0, plus a tail signal if count % cps != 0 after the loop" — two hand-maintained
 * spellings of one contract, where any drift (a reset in one but not the other, an off-by-one in
 * the tail condition) is a cross-device deadlock. Here both sides call @c advance() once per chunk
 * and read their own predicate; the invariant "waits issued == signals issued, at every prefix"
 * holds by construction because both predicates fold over the same counter.
 *
 * Usage — reader:  if (cadence.wait_due()) { wait(sem, ++target); }  cadence.advance();
 * Usage — writer:  cadence.advance();  if (cadence.signal_due()) { inc(sem); }
 *                  ... after the channel loop:  if (cadence.tail_due()) { inc(sem); }
 * Both: reset() at the same point the pre-migration kernels reset their counter (per slice
 * target / per final-reduction phase).
 */
class SyncCadence {
public:
    explicit SyncCadence(uint32_t chunks_per_sync) : cps_(chunks_per_sync) {}

    /// Reader side: the wait that admits the next @c cps_ chunks is due. Read BEFORE advance().
    bool wait_due() const { return count_ % cps_ == 0; }
    /// Both sides: account one chunk. Call exactly once per chunk.
    void advance() { ++count_; }
    /// Writer side: a signal is due for the chunk just advanced past. Read AFTER advance().
    bool signal_due() const { return count_ % cps_ == 0; }
    /// Writer side: chunks since the last signal exist and still need their signal (the tail of a
    /// walk whose chunk count chunks_per_sync does not divide). Read after the channel loop.
    bool tail_due() const { return count_ % cps_ != 0; }
    /// Restart the cadence (per slice target / per phase — mirror the paired kernel exactly).
    void reset() { count_ = 0; }

private:
    uint32_t cps_;
    uint32_t count_ = 0;
};

// ===========================================================================
// S7 — the LINE reduce-scatter schedule pieces
// ===========================================================================

/// First slice a line reduce-scatter worker processes: the far end of its direction's walk
/// (forward counts DOWN from ring_size-1; backward counts UP from 0). The walk visits
/// num_targets_in_direction slices and never leaves [0, ring_size), so there is no wrap.
constexpr uint32_t line_rs_first_slice(bool is_forward, uint32_t ring_size) { return is_forward ? ring_size - 1 : 0; }

/**
 * @brief Walks the line slice sequence one step per slice target. The line counterpart of
 *        RingSliceCursor — no wrap (a line walk stays inside [0, ring_size)), and the direction
 *        sign is inverted relative to the ring walk: FORWARD DECREMENTS, backward increments.
 *
 * Unlike the ring cursor there is no per-batch trap to warn about — the pre-migration line
 * kernels also re-seed per batch, and the same rule applies: construct (or reset()) INSIDE the
 * batch loop.
 */
class LineSliceCursor {
public:
    LineSliceCursor(bool is_forward, uint32_t ring_size) :
        slice_idx_(line_rs_first_slice(is_forward, ring_size)), is_forward_(is_forward), ring_size_(ring_size) {}

    /// Restart the walk at the first slice (per batch).
    void reset() { slice_idx_ = line_rs_first_slice(is_forward_, ring_size_); }

    /// The slice index for the current target. No wrap is needed or performed.
    uint32_t slice() const { return slice_idx_; }

    /// Step to the next slice target. Forward walks DOWN toward my_chip_id, backward UP.
    void advance() { slice_idx_ += is_forward_ ? -1 : 1; }

private:
    int32_t slice_idx_ = 0;
    bool is_forward_ = false;
    uint32_t ring_size_ = 0;
};

/// The final-reduction mode split shared by the line reader and writer — one definition of WHO
/// accumulates and WHO hands off, so the two sides of the fwd/bwd handshake cannot disagree:
/// when both directions land a final reduction on one core pair, the BACKWARD side ACCUMULATES
/// onto the output the forward side just wrote (reads the OUTPUT tensor instead of the input)...
constexpr bool line_rs_accumulate_output(bool sync_with_other_direction, bool is_forward) {
    return sync_with_other_direction && !is_forward;
}
/// ...and the FORWARD side hands each finished chunk to the backward reader (barrier + semaphore
/// inc per chunk). Exactly one of the pair holds in the synced case; neither holds unsynced.
constexpr bool line_rs_forward_hands_off(bool sync_with_other_direction, bool is_forward) {
    return sync_with_other_direction && is_forward;
}

/// Re-base a row offset expressed against one row stride onto another. The line final reduction
/// walks the OUTPUT tensor with the worker's start_row_offset, which the host expresses in
/// input-tensor rows (stride input_tensor_Wt); the output slice's row stride is slice_Wt.
constexpr uint32_t rebase_row_offset(uint32_t row_offset, uint32_t from_stride, uint32_t to_stride) {
    return row_offset / from_stride * to_stride;
}

/**
 * @brief The channel -> chunk walk one line-topology phase performs per slice target (and once
 *        more for the final reduction): channels x a plain [start, end) granularity walk.
 *
 * There is deliberately NO parity split and NO skip here: the ring schedule interleaves its two
 * directions over each slice's chunks (even/odd), while the line's two directions cover disjoint
 * SLICE SEQUENCES and each walks every chunk of its own slices. The compute kernel walks this too
 * (per reduction step), which is what keeps the three kernels' chunk boundaries — and therefore
 * their CB protocol — identical.
 *
 * Construct once; reset() re-arms for the next target / batch / phase. Nest exactly:
 * @code
 *   walk.reset();
 *   while (walk.next_channel()) {
 *       // per-channel: reset_offsets() on the walkers
 *       while (walk.next_chunk()) {
 *           const uint32_t n = walk.tiles_this_chunk();  // 1..granularity; short final chunk normal
 *           ...
 *       }
 *       // per-channel: bump_base() on the walkers
 *   }
 * @endcode
 */
class LineChannelWalk {
public:
    /**
     * @param channels             Per-phase channel count (slice_C).
     * @param tile_granularity     Maximum tiles per chunk (also the CB protocol granule).
     * @param start_tiles_read     This worker's first tile index within a channel.
     * @param start_tiles_to_read  This worker's end tile index within a channel (exclusive).
     */
    LineChannelWalk(
        uint32_t channels, uint32_t tile_granularity, uint32_t start_tiles_read, uint32_t start_tiles_to_read) :
        channels_(channels), granularity_(tile_granularity), start_(start_tiles_read), end_(start_tiles_to_read) {}

    /// Re-arm the walk for the next target / batch / phase.
    void reset() {
        channel_ = 0;
        channel_started_ = false;
    }

    /// Advance to the next channel, re-arming the chunk walk. False when channels are exhausted.
    bool next_channel() {
        if (channel_started_) {
            ++channel_;
        } else {
            channel_started_ = true;
        }
        if (channel_ >= channels_) {
            return false;
        }
        chunk_started_ = false;
        tiles_read_ = start_;
        tiles_this_chunk_ = 0;
        return true;
    }

    /// Advance to the next chunk of this channel. False when the tile range is exhausted.
    bool next_chunk() {
        if (chunk_started_) {
            tiles_read_ += tiles_this_chunk_;
        } else {
            chunk_started_ = true;
        }
        if (tiles_read_ >= end_) {
            return false;
        }
        tiles_this_chunk_ = std::min(end_ - tiles_read_, granularity_);
        return true;
    }

    uint32_t channel_idx() const { return channel_; }
    /// Tiles covered by the current chunk (1..granularity — a line chunk is never empty).
    uint32_t tiles_this_chunk() const { return tiles_this_chunk_; }

private:
    uint32_t channels_ = 0;
    uint32_t granularity_ = 0;
    uint32_t start_ = 0;
    uint32_t end_ = 0;

    uint32_t channel_ = 0;
    uint32_t tiles_read_ = 0;
    uint32_t tiles_this_chunk_ = 0;
    bool channel_started_ = false;
    bool chunk_started_ = false;
};

// ===========================================================================
// S8 — the DIM-ZERO RING reduce-scatter schedule pieces
// ===========================================================================

/// First slice a dim-zero ring worker processes: its nearest neighbour's, walking away from
/// itself (the dim-3 ring starts half-way across instead). May be -1; RingSliceCursor wraps it.
constexpr int32_t dim_zero_ring_first_slice(uint32_t my_chip_id, bool direction) {
    return direction ? static_cast<int32_t>(my_chip_id) - 1 : static_cast<int32_t>(my_chip_id) + 1;
}

/**
 * @brief The dim-zero ring family's batch -> chunk walk: the two directions interleave over each
 *        slice as alternating own/other chunk PAIRS — one worker takes the halved chunks, its
 *        opposite the full-width ones — with the opposite parity's chunk stepped over in the same
 *        call.
 *
 * This is the SAME chunk-size sequence the dim-3 RingRsSchedule emits (even = remaining/2, odd =
 * remaining, both capped at granularity), spelled the way the dim-zero kernels consume it, with
 * one PROTOCOL DIFFERENCE that is the reason this is a separate type: a ZERO-TILE own chunk still
 * runs the full CB protocol (reserve/push or wait/pop of one granule with no tiles touched),
 * where the dim-3 schedule's skip() elides the CBs entirely. The dim-zero reader, compute kernel
 * and writer all push/pop that empty granule in lockstep; eliding it on one side is a CB-wait
 * deadlock.
 *
 * Tile ids in this family are DENSE (base + position() + j) — there are no row walkers.
 *
 * Nest exactly (the reduction kernel drives the same walk with no ids):
 * @code
 *   walk.reset();                       // per ring step
 *   while (walk.next_batch()) {         // slice_B
 *       while (walk.next_chunk()) {     // one OWN chunk per call; opposite parity auto-stepped
 *           const uint32_t n = walk.tiles_this_chunk();   // 0..granularity; 0 still does CBs
 *           // ids: base + walk.position() + j for j in [0, n)
 *       }
 *       // per-batch: bump the dense base by batch_num_pages
 *   }
 * @endcode
 */
class DimZeroChunkWalk {
public:
    /**
     * @param batches          Inner batch count (slice_B — dim 0 scatters ON the batch dim).
     * @param tile_granularity Maximum tiles per chunk (also the CB protocol granule).
     * @param start_tiles_read This worker's first tile index within a batch.
     * @param start_tiles_to_read This worker's end tile index within a batch (exclusive).
     * @param direction        Ring direction; selects which half of the pair this worker owns
     *                         (direction takes the halved chunk, !direction the full-width one
     *                         after an initial half-step).
     */
    DimZeroChunkWalk(
        uint32_t batches,
        uint32_t tile_granularity,
        uint32_t start_tiles_read,
        uint32_t start_tiles_to_read,
        bool direction) :
        batches_(batches),
        granularity_(tile_granularity),
        start_(start_tiles_read),
        end_(start_tiles_to_read),
        direction_(direction) {}

    /// Re-arm the walk for the next ring step.
    void reset() {
        batch_ = 0;
        batch_started_ = false;
    }

    /// Advance to the next batch, re-arming the chunk walk (including the backward worker's
    /// initial step over the forward worker's first chunk). False when batches are exhausted.
    bool next_batch() {
        if (batch_started_) {
            ++batch_;
        } else {
            batch_started_ = true;
        }
        if (batch_ >= batches_) {
            return false;
        }
        tiles_read_ = start_;
        if (!direction_ && end_ > tiles_read_) {
            tiles_read_ += std::min((end_ - tiles_read_) / 2, granularity_);
        }
        chunk_started_ = false;
        tiles_this_chunk_ = 0;
        return true;
    }

    /// Advance to this worker's next own chunk, stepping over the opposite parity's chunk that
    /// follows it. False when the batch's tile range is exhausted.
    bool next_chunk() {
        if (chunk_started_) {
            tiles_read_ += tiles_this_chunk_;
            const uint32_t remaining = end_ > tiles_read_ ? end_ - tiles_read_ : 0;
            if (remaining > 0) {
                // Step over the opposite direction's chunk.
                tiles_read_ += direction_ ? std::min(remaining, granularity_) : std::min(remaining / 2, granularity_);
            }
        } else {
            chunk_started_ = true;
        }
        if (tiles_read_ >= end_) {
            return false;
        }
        const uint32_t remaining = end_ - tiles_read_;
        tiles_this_chunk_ = direction_ ? std::min(remaining / 2, granularity_) : std::min(remaining, granularity_);
        return true;
    }

    uint32_t batch_idx() const { return batch_; }
    /// Tiles in the current own chunk — ZERO IS LEGAL and still runs the full CB protocol.
    uint32_t tiles_this_chunk() const { return tiles_this_chunk_; }
    /// The current chunk's first tile index within [start, end) — the dense-id offset.
    uint32_t position() const { return tiles_read_; }

private:
    uint32_t batches_ = 0;
    uint32_t granularity_ = 0;
    uint32_t start_ = 0;
    uint32_t end_ = 0;
    bool direction_ = false;

    uint32_t batch_ = 0;
    uint32_t tiles_read_ = 0;
    uint32_t tiles_this_chunk_ = 0;
    bool batch_started_ = false;
    bool chunk_started_ = false;
};

}  // namespace ttnn::ccl::schedule
