// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Runtime chunk-size picker, shared by the reader, writer, and compute kernels.
//
// The kernel reads this expert's actual token count device-side and picks
// chunk_M_tiles (hence per_core_M = chunk_M_tiles / GRID_Y and num_chunks)
// FROM THAT COUNT — the adaptive per_core_M work that used to be sized on the
// host from an `expected_tokens` argument now happens at runtime, so no caller
// has to know the token count in advance.
//
// CBs are sized on the host to the compile-time MAX shape (per_core_M = 8,
// chunk_M_tiles = 64; a big-model L1 guard may lower the max, passed in as
// `max_chunk`). The picker never returns more than `max_chunk`, so the runtime
// per_core_M always fits the allocated CBs; a smaller pick simply uses fewer of
// the reserved tiles and shrinks every M-dimension loop.
//
// The three kernels MUST derive identical chunk_M_tiles / per_core_M / chunk
// counts from the same count, or the reader/compute/writer disagree on the row
// mapping and FFN output lands on the wrong token rows. This is the single
// source of truth. Pure integer arithmetic (no NoC/CB/reg APIs) so the header
// is valid in BRISC, NCRISC, and TRISC translation units alike.
namespace adaptive_chunk {

constexpr uint32_t kGridY = 8;  // M-row cores; a chunk spans per_core_M * kGridY tile-rows

// Chunk layout for `count_tiles` tile-rows, given the CB-sized maximum chunk
// `max_chunk` (= per_core_M_max * kGridY): a run of FULL chunks of max_chunk,
// then ONE tail chunk sized down to the remainder. This minimizes the chunk
// count (each chunk re-reads the full gate/up/down weights, the DRAM bottleneck)
// AND minimizes phantom M-work (the tail covers only the leftover rows), so
// per_core_M adapts to the actual token count per chunk with essentially no
// wasted rows — e.g. 160 tiles -> 64 + 64 + 32 (per_core_M 8,8,4), 3 chunks,
// zero phantom.
//
// CRITICAL — per_core_M BOUNDS WORK, IT DOES NOT SIZE CB BLOCKS.
//
// per_core_M is free to differ per expert and per chunk (that is the whole point:
// each expert gets the chunk shape its own token count deserves). It bounds MACs,
// tilize strips, DRAM reads, multicast payloads and emitted output rows.
//
// It must NOT size any circular-buffer block. cb_push_back / cb_pop_front wrap the
// FIFO pointer only when it lands EXACTLY on fifo_limit:
//
//     fifo_wr_ptr += num_words;
//     // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
//     ASSERT(fifo_wr_ptr <= fifo_limit);
//     if (fifo_wr_ptr == fifo_limit) { fifo_wr_ptr -= fifo_size; }
//                                              (tt_metal/hw/inc/api/dataflow/dataflow_api.h)
//
// A block size that changes between pushes leaves the pointer at an offset the
// next (larger) block does not divide; that push OVERSHOOTS fifo_limit, the
// equality fails, the wrap is skipped, the ASSERT is a no-op in release, and the
// CB pointer then runs away into neighbouring L1 for the rest of the kernel.
//
// Under the retired one-program-per-expert design this was invisible: every expert
// launched its own program, so the pointers restarted at fifo_addr and only the
// within-expert full-chunks-then-tail order mattered (and that order happens to
// realign). ONE program looping over all local experts carries the pointers over,
// so a per-expert block size corrupts L1 — e.g. on cb_activated (ring = 8 blocks at
// per_core_M=1, one push per chunk) a run of per_core_M=1 experts leaves the
// pointer at 3, and a following per_core_M=2 expert pushes 3 -> 5 -> 7 -> 9, which
// never equals 8.
//
// So the reader/compute kernels reserve/push/wait/pop a CONSTANT compile-time-max
// block on cb_x_rm, cb_in0_x, cb_gate_intermed, cb_activated and cb_mm_partials_*,
// and settle the runtime remainder with O(1) pointer-only bumps. This is the same
// pattern cb_in0_down_full and cb_out already use. The adaptive win is untouched:
// no MAC, tilize, DRAM read or multicast byte is spent on the padded rows.
//
// The tail per_core_M is still returned as a DIVISOR of per_core_M_max, so the
// runtime rows always tile evenly inside the constant block.

// Clamp a DEVICE-PROVIDED token-tile count to the capacity this program was
// built for: `num_chunks_max` chunks of at most `max_chunk` tile-rows each.
//
// counts[] is produced on device (dispatch) and is never host-validated, so an
// over-capacity entry must not be allowed to drive the chunk loop past
// num_chunks_max: the CBs, the num_chunks compile-time arg and the output
// buffer are all sized to that bound, and running past it reads/writes outside
// this expert's region. ASSERT is a no-op unless watcher / lightweight kernel
// asserts are enabled, so the bound has to be enforced by arithmetic to hold in
// Release builds — the kernels ASSERT on top of it to still fail loudly in
// debug builds.
//
// Clamping the COUNT (not the chunk index) is what keeps the three kernels in
// lockstep: reader, compute and writer all derive effective_chunks, per_core_M
// and their row-validity guards from this same value, so they agree on the row
// mapping and the excess rows are uniformly dropped rather than emitted at the
// wrong offsets.
inline uint32_t clamp_count_tiles(uint32_t count_tiles, uint32_t max_chunk, uint32_t num_chunks_max) {
    const uint32_t capacity_tiles = num_chunks_max * max_chunk;
    return (count_tiles < capacity_tiles) ? count_tiles : capacity_tiles;
}

// Number of chunks for `count_tiles`: full chunks of max_chunk + one tail chunk.
inline uint32_t num_chunks(uint32_t count_tiles, uint32_t max_chunk) {
    if (count_tiles < 1) {
        return 0;
    }
    const uint32_t num_full = count_tiles / max_chunk;
    const uint32_t tail = count_tiles - num_full * max_chunk;
    return num_full + ((tail > 0) ? 1u : 0u);
}

// per_core_M for chunk index `c`: per_core_M_max for the full chunks; for the
// tail chunk, the smallest DIVISOR of per_core_M_max whose *kGridY covers the
// tail tiles (so the tail does the least M-work while its block still tiles
// evenly into the CBs).
inline uint32_t per_core_M_for_chunk(uint32_t c, uint32_t count_tiles, uint32_t max_chunk) {
    const uint32_t per_core_M_max = max_chunk / kGridY;
    const uint32_t num_full = count_tiles / max_chunk;
    if (c < num_full) {
        return per_core_M_max;
    }
    const uint32_t tail = count_tiles - num_full * max_chunk;  // > 0 for the tail chunk
    uint32_t need = (tail + kGridY - 1) / kGridY;              // min rows/core to cover the tail
    if (need < 1) {
        need = 1;
    }
    for (uint32_t d = need; d <= per_core_M_max; ++d) {
        if ((per_core_M_max % d) == 0) {
            return d;
        }
    }
    return per_core_M_max;
}

}  // namespace adaptive_chunk
