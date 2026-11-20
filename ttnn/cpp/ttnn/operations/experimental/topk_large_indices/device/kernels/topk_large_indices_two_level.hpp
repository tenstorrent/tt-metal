// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the three kernels of a two-level program. A two-level program selects each row's top k from the
// maxima of 32-key blocks first, then from the keys of the 2048 selected blocks only:
//   stage 1 (phase 1, every core): block maxima of all rows, claimed by the cores in 32-tile units (reader: tiles;
//            compute: tilize + max reduce; writer: maxima to the maxima tensor, then a count on the row's owner);
//   stage 2 (each core its rows, as soon as a row's maxima are complete, between phase-1 units): top-2048 of the
//            row's maxima -> selected block ids (compute), ascending block table (writer);
//   stage 3: the selected blocks gathered into one 65,536-key candidate row (reader), its exact top-2048
//            (compute), candidate positions mapped back to keys (writer).
// The program decides once, from the search length, whether its rows run this way or the single-level way. The
// two-level code compiles only with TOPK_TWO_LEVEL defined, so single-level programs build exactly as before.

#include <cstdint>

namespace topk_two_level {

constexpr uint32_t block_keys = 32;
constexpr uint32_t block_bytes = block_keys * sizeof(uint16_t);
constexpr uint32_t selected_blocks = 2048;
constexpr uint32_t candidate_keys = selected_blocks * block_keys;
// Keys per stage-1 tile: a 32x32 tile whose rows are 32 consecutive keys.
constexpr uint32_t tile_keys = 1024;
constexpr uint32_t sentinel = 0xFFFFFFFFu;

// The reader-to-compute mailbox of a two-level program: the single-level bounds (whole row, split half), then
// the mode and the search length.
constexpr uint32_t mailbox_mode_word = 4;
constexpr uint32_t mailbox_search_length_word = 5;
// The reader-to-writer mailbox page.
constexpr uint32_t writer_mailbox_mode_word = 0;
constexpr uint32_t writer_mailbox_search_length_word = 1;

constexpr uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

#ifdef KERNEL_BUILD
// Waits for this RISC's stores to L1, the last of them to the word at `addr`, to land before a CB push, the NoC or
// another core reads them: the fence orders the stores, a load from the address returns after the store, and an
// instruction using the loaded value waits for the load.
inline __attribute__((always_inline)) void wait_l1_store(uint32_t addr) {
    asm volatile("fence" ::: "memory");
    uint32_t value = *reinterpret_cast<const volatile uint32_t*>(addr & ~3u);
    asm volatile("mv %0, %0" : "+r"(value) : : "memory");
}
#endif

// The writer's stage-2 block flags: a byte per block, scanned four 32-bit words at a time.
constexpr uint32_t flag_words(uint32_t search_len) { return div_up(div_up(search_len, block_keys), 16) * 4; }

// The phase-1 units: 32-tile batches of the rows' keys. Cores claim them one at a time from a counter on the
// coordinator, so a core that streams faster does more of them. Units are numbered in waves of wave_rows rows,
// across the rows first within a wave: each input row is one DRAM page in one bank, so the units in flight at once
// spread over every bank, while each wave's rows finish early enough for their owners to start on them.
constexpr uint32_t wave_rows = 16;
// Which row the waves take as their o-th: in row order (the split rows last), or with split_first the split rows (the
// last split_rows rows) first.
struct Units {
    uint32_t total;
    uint32_t num_rows;
    uint32_t per_row;
    uint32_t split_rows;
    bool split_first;

    constexpr uint32_t ordered_row(uint32_t o) const {
        if (!split_first) {
            return o;
        }
        return o < split_rows ? num_rows - split_rows + o : o - split_rows;
    }
    constexpr uint32_t row(uint32_t unit) const {
        const uint32_t wave = unit / (wave_rows * per_row);
        const uint32_t rows = num_rows - wave * wave_rows < wave_rows ? num_rows - wave * wave_rows : wave_rows;
        return ordered_row(wave * wave_rows + (unit - wave * wave_rows * per_row) % rows);
    }
    constexpr uint32_t part(uint32_t unit) const {
        const uint32_t wave = unit / (wave_rows * per_row);
        const uint32_t rows = num_rows - wave * wave_rows < wave_rows ? num_rows - wave * wave_rows : wave_rows;
        return (unit - wave * wave_rows * per_row) / rows;
    }

    static constexpr Units of(
        uint32_t search_len,
        uint32_t batch_tiles,
        uint32_t num_rows,
        uint32_t split_rows = 0,
        bool split_first = false) {
        const uint32_t per_row = div_up(search_len, batch_tiles * tile_keys);
        return {per_row * num_rows, num_rows, per_row, split_rows, split_first};
    }
};

// The phase-1 schedule of a program whose split rows run their halves early (tl_early_halves), by search length. Below
// split_first_below the split rows' units come first, so their halves of stage 3 run during phase 1, and at most
// row_cap_fifths / 5 of the cores sort rows while units remain, so phase 1 keeps going. From split_first_below up,
// phase 1 is long enough to hide the whole rows, and a split row is the shortest work to leave for after it: its
// units come last, without a cap. (Measured on 108 cores, 160 rows: split-first gains to 256K, loses from 512K.)
constexpr uint32_t split_first_below = 393216;
constexpr uint32_t row_cap_fifths = 3;
constexpr bool split_rows_first(bool early_halves, uint32_t search_len) {
    return early_halves && search_len < split_first_below;
}

// Row ownership, like the single-level op's split: the first num_rows - split_rows rows are whole rows, row r on core
// r % num_cores as its local row r / num_cores (a whole number of rounds when there are split rows; without, the
// last round may be partial). The last split_rows rows are halved across core pairs (as the single-level assignment
// pairs them): split row whole_rows + u belongs to core 2u, which runs its stage 2 and the first half of its stage 3
// and merges in the second half that core 2u + 1 sorts. A core's row semaphore counts the finished phase-1 units of
// each local row in its own byte (a row has at most 64 units); a row is ready when its count reaches per_row.
constexpr uint32_t max_local_rows = 4;
constexpr uint32_t ready_increment(uint32_t local_row) { return 1u << (8 * local_row); }
constexpr uint32_t ready_count(uint32_t word, uint32_t local_row) { return (word >> (8 * local_row)) & 0xFFu; }
struct RowOwner {
    uint32_t core;
    uint32_t local_row;
};
constexpr RowOwner row_owner(uint32_t row, uint32_t num_rows, uint32_t split_rows, uint32_t num_cores) {
    const uint32_t whole_rows = num_rows - split_rows;
    if (row < whole_rows) {
        return {row % num_cores, row / num_cores};
    }
    return {2 * (row - whole_rows), whole_rows / num_cores};
}
// The block table goes from the writer to the reader a candidate chunk's blocks at a time: 2048 keys, 64 blocks.
constexpr uint32_t table_page_entries = 2048 / block_keys;
constexpr uint32_t table_page_bytes = table_page_entries * sizeof(uint32_t);
constexpr uint32_t table_pages = selected_blocks / table_page_entries;

// What the reader sends the compute (a header page in the stage-1 CB, then 32 tile pages) and the writer (its
// mailbox) next: a phase-1 unit's number, one of this core's rows, or the end.
constexpr uint32_t header_half = 0xFFFFFFFCu;   // this core's half of its split row's stage 3 (tl_early_halves)
constexpr uint32_t header_split = 0xFFFFFFFDu;  // a split row's stage 2; its halves of stage 3 follow the end, or with
                                                // tl_early_halves run as header_half items as soon as they can
constexpr uint32_t header_row = 0xFFFFFFFEu;
constexpr uint32_t header_end = 0xFFFFFFFFu;
constexpr uint32_t writer_header_code_word = 0;
constexpr uint32_t writer_header_row_word = 1;

// Whether a program runs two-level for this search length. An empty row (a real-token end of 0) runs two-level
// too: with no blocks every stage passes -inf through, so the row comes out as sentinels.
constexpr bool two_level_mode_for(uint32_t search_len, uint32_t cutover) {
    return search_len >= cutover || search_len == 0;
}

// Stage-2 chunks of 2048 maxima for a search length: at least two, so every stage runs on the split body. Maxima
// past the row's blocks are -inf.
constexpr uint32_t stage2_chunks(uint32_t search_len) {
    const uint32_t chunks = div_up(div_up(search_len, block_keys), 2048);
    return chunks < 2 ? 2 : chunks;
}

}  // namespace topk_two_level
