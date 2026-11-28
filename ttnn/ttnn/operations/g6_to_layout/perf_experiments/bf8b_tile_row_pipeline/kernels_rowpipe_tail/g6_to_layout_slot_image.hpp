// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"

// build_slot_image of same_bank_slot_writes, shared by the reader and the writer, which split a
// block's rows between them.
//
// An interleaved bank holds pages p, p + B, p + 2B, ... (B = contiguous_page_stride()) in
// consecutive aligned slots. A block's rows are output pages [first_page, first_page + rows)
// (pages are contiguous across tile-rows and images), so its slot image holds them run by run:
// run j (pages first_page + j + k * B) starts at slot j * q + min(j, m) and holds q + 1 slots for
// j < m, q after (q = rows / B, m = rows % B). Each run is then one contiguous NoC write.
//
// A row moves either with RISC word loads/stores of its live prefix (2 * W bytes rounded up to
// words) or with one self-aimed NoC read of it (rounded up to the L1 granule). The two RISCs' loads
// and stores contend for L1 (RISC copies split between them run no faster in total than on one); a
// NoC copy goes through the NIU instead, so the writer's share runs as NoC copies beside the
// reader's RISC copies. Either copy ends inside the slot (2 * W < slot); the rest of a slot is its
// page's alignment gap: the output buffer's own memory, never part of a logical value.

namespace g6_to_layout {

template <uint32_t slot_bytes>
struct SlotRuns {
    uint32_t first_page;
    uint32_t rows;
    uint32_t banks;
    uint32_t q;
    uint32_t m;

    FORCE_INLINE SlotRuns(const Block& block, uint32_t banks) :
        first_page(first_row_page_of(block.tile_row)),
        rows(first_row_page_of(block.tile_row + block.rows) - first_page),
        banks(banks),
        q(rows / banks),
        m(rows - q * banks) {}

    FORCE_INLINE uint32_t count() const { return rows < banks ? rows : banks; }
    FORCE_INLINE uint32_t slots(uint32_t j) const { return j < m ? q + 1 : q; }
    FORCE_INLINE uint32_t offset(uint32_t j) const { return (j * q + (j < m ? j : m)) * slot_bytes; }
    // Rows [0, reader_rows) are the reader's share of the build, the rest the writer's.
    FORCE_INLINE uint32_t reader_rows(uint32_t reader_eighths) const { return rows * reader_eighths / 8; }
};

// Slot of each successive row in a block's image, starting at row `row`.
template <uint32_t slot_bytes>
struct SlotCursor {
    uint32_t image;
    uint32_t banks;
    uint32_t m;
    uint32_t long_run_bytes;
    uint32_t short_run_bytes;
    uint32_t j;     // the row's run
    uint32_t run;   // first slot of run j
    uint32_t step;  // the row's slot inside its run, in bytes

    FORCE_INLINE SlotCursor(const SlotRuns<slot_bytes>& runs, uint32_t image, uint32_t row) :
        image(image),
        banks(runs.banks),
        m(runs.m),
        long_run_bytes((runs.q + 1) * slot_bytes),
        short_run_bytes(runs.q * slot_bytes),
        j(row % runs.banks),
        run(image + runs.offset(j)),
        step(row / runs.banks * slot_bytes) {}

    FORCE_INLINE uint32_t next() {
        const uint32_t dst = run + step;
        run += j < m ? long_run_bytes : short_run_bytes;
        if (++j == banks) {
            j = 0;
            run = image;
            step += slot_bytes;
        }
        return dst;
    }
};

// RISC fence plus compiler barrier: the face page was landed by the NoC, and the image is read by
// the NoC (the writer's runs) after the plain stores below.
FORCE_INLINE void slot_image_fence() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
}

constexpr uint32_t slot_copy_words = (Geometry::width * DATUM_BYTES + 3) / 4;
constexpr uint32_t slot_copy_noc_bytes =
    (Geometry::width * DATUM_BYTES + L1_ALIGNMENT - 1) / L1_ALIGNMENT * L1_ALIGNMENT;
// Rows per batch: every load of a batch issues before its stores; at most 16 words in registers.
constexpr uint32_t slot_copy_batch = slot_copy_words <= 1   ? 16
                                     : slot_copy_words <= 2 ? 8
                                     : slot_copy_words <= 4 ? 4
                                                            : 2;

// Copies the live prefix of `rows` consecutive face rows from `src` to the cursor's next slots.
// Consecutive rows land in different runs, so a batch's stores spread over L1 banks.
template <uint32_t rows, typename Cursor>
FORCE_INLINE void copy_row_prefixes(uint32_t src, Cursor& cursor) {
    uint32_t words[rows][slot_copy_words];
    const auto* face_rows = reinterpret_cast<const tt_l1_ptr uint32_t*>(src);
#pragma GCC unroll 16
    for (uint32_t b = 0; b < rows; ++b) {
#pragma GCC unroll 8
        for (uint32_t w = 0; w < slot_copy_words; ++w) {
            words[b][w] = face_rows[b * (FACE_ROW_BYTES / 4) + w];
        }
    }
#pragma GCC unroll 16
    for (uint32_t b = 0; b < rows; ++b) {
        auto* slot = reinterpret_cast<tt_l1_ptr uint32_t*>(cursor.next());
#pragma GCC unroll 8
        for (uint32_t w = 0; w < slot_copy_words; ++w) {
            slot[w] = words[b][w];
        }
    }
}

template <bool noc_copies, typename Cursor>
FORCE_INLINE void copy_face_rows(Noc noc, uint32_t src, uint32_t rows, Cursor& cursor) {
    if constexpr (noc_copies) {
        for (; rows > 0; --rows) {
            dataflow_kernel_lib::read_with_state<slot_copy_noc_bytes>(noc, cursor.next(), src);
            src += FACE_ROW_BYTES;
        }
        return;
    }
    for (; rows >= slot_copy_batch; rows -= slot_copy_batch) {
        copy_row_prefixes<slot_copy_batch>(src, cursor);
        src += slot_copy_batch * FACE_ROW_BYTES;
    }
    for (; rows > 0; --rows) {
        copy_row_prefixes<1>(src, cursor);
        src += FACE_ROW_BYTES;
    }
}

// Copies rows [row_begin, row_end) of `block` (in page order) from its face page to their slots in
// `image`. The face page holds tile i's faces 0 and 2 at i * LEFT_SLOT_BYTES, so row r of tile-row
// i sits at i * LEFT_SLOT_BYTES + r * FACE_ROW_BYTES. With noc_copies every copy has landed when
// this returns: it waits for all of this RISC's NoC reads, so none other may be in flight.
template <uint32_t slot_bytes, bool noc_copies>
FORCE_INLINE void build_slot_rows(
    const Block& block,
    const SlotRuns<slot_bytes>& runs,
    uint32_t face_page,
    uint32_t image,
    uint32_t row_begin,
    uint32_t row_end) {
    if (row_begin >= row_end) {
        return;
    }
    Noc noc;
    if constexpr (noc_copies) {
        dataflow_kernel_lib::set_read_state<slot_copy_noc_bytes>(noc, face_page);
    } else {
        slot_image_fence();
    }
    SlotCursor<slot_bytes> cursor(runs, image, row_begin);
    if (runs.rows == block.rows * TILE_HW) {
        // Every tile-row is full: row i sits at face_page + i * FACE_ROW_BYTES.
        copy_face_rows<noc_copies>(noc, face_page + row_begin * FACE_ROW_BYTES, row_end - row_begin, cursor);
    } else {
        // Some tile-row is short (H % 32 != 0 or H < 32): take each tile-row's part of the range.
        uint32_t row = 0;
        uint32_t tile = face_page;
        for (uint32_t i = 0; i < block.rows && row < row_end; ++i) {
            const uint32_t live_rows = live_rows_of(block.tile_row + i);
            const uint32_t lo = row_begin > row ? row_begin - row : 0;
            const uint32_t hi = min_u32(live_rows, row_end - row);
            if (lo < hi) {
                copy_face_rows<noc_copies>(noc, tile + lo * FACE_ROW_BYTES, hi - lo, cursor);
            }
            row += live_rows;
            tile += LEFT_SLOT_BYTES;
        }
    }
    if constexpr (noc_copies) {
        noc_async_read_barrier();
    } else {
        slot_image_fence();
    }
}

}  // namespace g6_to_layout
