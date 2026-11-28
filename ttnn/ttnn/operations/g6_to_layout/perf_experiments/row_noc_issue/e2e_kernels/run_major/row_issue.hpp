// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-row NoC issue of interleaved ROW_MAJOR pages: N consecutive pages, one transaction per page,
// a byte segment [offset, offset + bytes) of each, from / to an L1 stage with a fixed row stride.
//
// Variants (compile-time, identical transactions; only how each one is issued differs):
//   ISSUE_ACCESSOR    TensorAccessor::get_noc_addr(page, offset) + noc_async_write / noc_async_read
//                     per row (a divide by the bank count, two bank-table loads and the full 6-register
//                     command per row).
//   ISSUE_CURSOR      the page's bank and bank slot are stepped (bank + 1, slot + aligned page on wrap)
//                     instead of divided; same noc_async_write / noc_async_read call per row.
//   ISSUE_STATE       cursor + stateful command buffer: CTRL / LEN (and MID) are programmed once per
//                     call with *_one_packet_set_state; per row only the bank coordinate, the two
//                     local addresses and CMD_CTRL are written. RAW: there is no dataflow API that
//                     changes only the coordinate of a stateful command, so the coordinate register is
//                     written directly (NOC_CMD_BUF_WRITE_REG after the cmd_buf_ready poll) and the
//                     issue is ncrisc_noc_{write,read}_with_state<..., one_packet, skip_cmdbuf_chk>.
//   ISSUE_BANK_MAJOR  public API only: rows are issued bank by bank (page p, p + NB, p + 2 NB, ...);
//                     *_one_packet_set_state once per bank, *_one_packet_with_state per row. Only
//                     fewer coordinate writes than ISSUE_STATE when a call holds > 1 row per bank
//                     (DRAM: 8 banks, 32 rows -> 4 rows per bank; L1: 110 banks -> 1).
//   ISSUE_STATE_BULK  ISSUE_STATE without the per-row L1 counter updates: the issued / acked counters
//                     are bumped once per call (before any flush or barrier reads them).
// Segments above NOC_MAX_BURST_SIZE fall back to ISSUE_CURSOR (one packet is the stateful contract),
// and so does a call of one row (H = 1 images): programming the state would cost more than it saves.
//
// Every variant works on a plain row-major stage: the source / destination L1 address of each row
// is free, so no variant needs the rows of one bank to be adjacent in L1. Only coalescing one bank's
// pages into one transaction (ISSUE_RUN_COALESCE, writes only, row_copy.cpp) needs a run-major stage.

#include "api/dataflow/dataflow_api.h"

namespace row_issue {

constexpr uint32_t ISSUE_ACCESSOR = 0;
constexpr uint32_t ISSUE_CURSOR = 1;
constexpr uint32_t ISSUE_STATE = 2;
constexpr uint32_t ISSUE_BANK_MAJOR = 3;
constexpr uint32_t ISSUE_STATE_BULK = 4;
// Diagnostics only: ISSUE_STATE_BULK with a nop delay after each row (pacing the issue rate).
constexpr uint32_t ISSUE_STATE_BULK_PACE8 = 6;
constexpr uint32_t ISSUE_STATE_BULK_PACE4 = 7;

// Diagnostics only: ISSUE_STATE_BULK with a wall-clock floor between consecutive rows.
constexpr uint32_t ISSUE_STATE_BULK_FLOOR48 = 8;
constexpr uint32_t ISSUE_STATE_BULK_FLOOR72 = 9;
constexpr uint32_t ISSUE_STATE_BULK_FLOOR96 = 10;

template <uint32_t variant>
FORCE_INLINE void pace() {
    constexpr uint32_t floor = variant == ISSUE_STATE_BULK_FLOOR48   ? 48
                               : variant == ISSUE_STATE_BULK_FLOOR72 ? 72
                               : variant == ISSUE_STATE_BULK_FLOOR96 ? 96
                                                                     : 0;
    if constexpr (floor != 0) {
        static uint32_t last = 0;
        uint32_t now;
        do {
            now = get_timestamp_32b();
        } while (now - last < floor);
        last = now;
    }
    constexpr uint32_t n = variant == ISSUE_STATE_BULK_PACE8 ? 8 : variant == ISSUE_STATE_BULK_PACE4 ? 4 : 0;
#pragma GCC unroll 8
    for (uint32_t i = 0; i < n; ++i) {
        asm volatile("nop");
    }
}

// Consecutive interleaved pages: bank = page % NB walks the banks in order; the bank-local slot of
// the page advances by one aligned page each time the bank index wraps.
template <bool DRAM>
struct PageCursor {
    static constexpr uint32_t NB = DRAM ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    uint32_t bank;
    uint32_t slot;  // bank_base + (page / NB) * aligned_page + byte offset (without the bank's own offset)
    uint32_t aligned_page;

    FORCE_INLINE void init(uint32_t bank_base, uint32_t aligned, uint32_t page, uint32_t offset) {
        const uint32_t q = interleaved_addr_gen::get_bank_offset_index<DRAM>(page);
        bank = interleaved_addr_gen::get_bank_index<DRAM>(page, q);
        slot = bank_base + q * aligned + offset;
        aligned_page = aligned;
    }
    FORCE_INLINE void step() {
        if (++bank == NB) {
            bank = 0;
            slot += aligned_page;
        }
    }
    FORCE_INLINE uint32_t xy(uint8_t noc) const { return interleaved_addr_gen::get_noc_xy<DRAM>(bank, noc); }
    FORCE_INLINE uint32_t local() const { return slot + interleaved_addr_gen::get_bank_offset<DRAM>(bank); }
    FORCE_INLINE uint64_t noc_addr(uint8_t noc) const {
        return noc_address_backend::bank_address<DRAM>(bank, local(), noc);
    }
};

template <typename Accessor>
FORCE_INLINE PageCursor<Accessor::is_dram> cursor_at(const Accessor& acc, uint32_t page, uint32_t offset) {
    PageCursor<Accessor::is_dram> c;
    c.init(acc.bank_base_address, acc.get_aligned_page_size(), page, offset);
    return c;
}

// ---------------------------------------------------------------- writes (L1 stage -> pages)
// A sharded accessor has no interleaved bank walk: it always takes the accessor path.
template <uint32_t requested, typename Accessor>
constexpr uint32_t effective_variant = Accessor::DSpec::is_interleaved ? requested : ISSUE_ACCESSOR;

template <uint32_t requested, typename Accessor>
FORCE_INLINE void write_rows(
    const Accessor& out,
    uint32_t first_page,
    uint32_t rows,
    uint32_t offset,
    uint32_t bytes,
    uint32_t src,
    uint32_t src_stride) {
    constexpr uint32_t variant = effective_variant<requested, Accessor>;
    constexpr bool DRAM = Accessor::is_dram;
    const uint8_t noc = noc_index;
    if constexpr (variant == ISSUE_ACCESSOR) {
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write(src, out.get_noc_addr(first_page + r, offset), bytes);
            src += src_stride;
        }
    } else if constexpr (variant == ISSUE_BANK_MAJOR) {
        constexpr uint32_t NB = PageCursor<DRAM>::NB;
        if (bytes > NOC_MAX_BURST_SIZE) {
            return write_rows<ISSUE_CURSOR>(out, first_page, rows, offset, bytes, src, src_stride);
        }
        const uint32_t phases = rows < NB ? rows : NB;
        auto c = cursor_at(out, first_page, offset);
        const uint32_t bank_src_stride = NB * src_stride;
        for (uint32_t j = 0; j < phases; ++j) {
            noc_async_write_one_packet_set_state(c.noc_addr(noc), bytes, noc);
            uint32_t dst = c.local();
            for (uint32_t r = j; r < rows; r += NB) {
                noc_async_write_one_packet_with_state(src + (r - j) / NB * bank_src_stride, dst, noc);
                dst += c.aligned_page;
            }
            src += src_stride;
            c.step();
        }
    } else {
        auto c = cursor_at(out, first_page, offset);
        if (variant == ISSUE_CURSOR || bytes > NOC_MAX_BURST_SIZE || rows < 2) {
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_write(src, c.noc_addr(noc), bytes);
                src += src_stride;
                c.step();
            }
            return;
        }
        // Dynamic-NoC mode keeps its counters elsewhere: bump them per row there.
        constexpr bool per_row_counter = variant == ISSUE_STATE || noc_mode == DM_DYNAMIC_NOC;
        noc_async_write_one_packet_set_state(c.noc_addr(noc), bytes, noc);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t xy = c.xy(noc);
            const uint32_t dst = c.local();
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, xy);
            ncrisc_noc_write_with_state<noc_mode, false, per_row_counter, true, true>(noc, write_cmd_buf, src, dst);
            pace<variant>();
            src += src_stride;
            c.step();
        }
        if constexpr (!per_row_counter) {
            noc_nonposted_writes_num_issued[noc] += rows;
            noc_nonposted_writes_acked[noc] += rows;
        }
    }
}

// ---------------------------------------------------------------- reads (pages -> L1 stage)
template <uint32_t requested, typename Accessor>
FORCE_INLINE void read_rows(
    const Accessor& in,
    uint32_t first_page,
    uint32_t rows,
    uint32_t offset,
    uint32_t bytes,
    uint32_t dst,
    uint32_t dst_stride) {
    constexpr uint32_t variant = effective_variant<requested, Accessor>;
    constexpr bool DRAM = Accessor::is_dram;
    const uint8_t noc = noc_index;
    if constexpr (variant == ISSUE_ACCESSOR) {
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(in.get_noc_addr(first_page + r, offset), dst, bytes);
            dst += dst_stride;
        }
    } else if constexpr (variant == ISSUE_BANK_MAJOR) {
        constexpr uint32_t NB = PageCursor<DRAM>::NB;
        if (bytes > NOC_MAX_BURST_SIZE) {
            return read_rows<ISSUE_CURSOR>(in, first_page, rows, offset, bytes, dst, dst_stride);
        }
        const uint32_t phases = rows < NB ? rows : NB;
        auto c = cursor_at(in, first_page, offset);
        const uint32_t bank_dst_stride = NB * dst_stride;
        for (uint32_t j = 0; j < phases; ++j) {
            noc_async_read_one_packet_set_state(c.noc_addr(noc), bytes, 0, noc);
            uint32_t src = c.local();
            for (uint32_t r = j; r < rows; r += NB) {
                noc_async_read_one_packet_with_state(src, dst + (r - j) / NB * bank_dst_stride, 0, noc);
                src += c.aligned_page;
            }
            dst += dst_stride;
            c.step();
        }
    } else {
        auto c = cursor_at(in, first_page, offset);
        if (variant == ISSUE_CURSOR || bytes > NOC_MAX_BURST_SIZE || rows < 2) {
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(c.noc_addr(noc), dst, bytes);
                dst += dst_stride;
                c.step();
            }
            return;
        }
        // Dynamic-NoC mode keeps its counters elsewhere: bump them per row there.
        constexpr bool per_row_counter = variant == ISSUE_STATE || noc_mode == DM_DYNAMIC_NOC;
        noc_async_read_one_packet_set_state(c.noc_addr(noc), bytes, 0, noc);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t xy = c.xy(noc);
            const uint32_t src = c.local();
            while (!noc_cmd_buf_ready(noc, read_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, xy);
            ncrisc_noc_read_with_state<noc_mode, per_row_counter, true, true>(noc, read_cmd_buf, src, dst);
            pace<variant>();
            dst += dst_stride;
            c.step();
        }
        if constexpr (!per_row_counter) {
            noc_reads_num_issued[noc] += rows;
        }
    }
}

// ---------------------------------------------------------------- run-major stage (option 4)
// Row r of a tile-row of `rows` consecutive pages sits at run slot run_pos(r): the rows of one DRAM
// bank (r, r + NB, r + 2 NB, ...) are adjacent, phase j = r % NB first. With a full-row segment and a
// stage row stride equal to the output's aligned page, phase j is then ONE write of count_j pages
// into consecutive slots of its bank. Needs the producer to lay the stage out this way.
FORCE_INLINE uint32_t run_phase_start(uint32_t j, uint32_t rows) {
    const uint32_t per = rows / NUM_DRAM_BANKS;
    const uint32_t extra = rows % NUM_DRAM_BANKS;
    return j * per + (j < extra ? j : extra);
}
FORCE_INLINE uint32_t run_pos(uint32_t r, uint32_t rows) {
    return run_phase_start(r % NUM_DRAM_BANKS, rows) + r / NUM_DRAM_BANKS;
}

// Writes one tile-row from a run-major stage: one write per DRAM bank phase when the output is
// interleaved DRAM with aligned page == stage_row (and the segment is the whole row); otherwise one
// write per row from its run slot.
template <typename Accessor>
FORCE_INLINE void write_run_major_rows(
    const Accessor& out, uint32_t first_page, uint32_t rows, uint32_t bytes, uint32_t src, uint32_t stage_row) {
    if constexpr (Accessor::DSpec::is_interleaved && Accessor::is_dram) {
        if (out.get_aligned_page_size() == stage_row) {
            const uint32_t phases = rows < NUM_DRAM_BANKS ? rows : NUM_DRAM_BANKS;
            const uint32_t per = rows / NUM_DRAM_BANKS;
            const uint32_t extra = rows % NUM_DRAM_BANKS;
            for (uint32_t j = 0; j < phases; ++j) {
                const uint32_t count = per + (j < extra ? 1 : 0);
                noc_async_write(
                    src + run_phase_start(j, rows) * stage_row, out.get_noc_addr(first_page + j, 0), count * stage_row);
            }
            return;
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write(src + run_pos(r, rows) * stage_row, out.get_noc_addr(first_page + r, 0), bytes);
    }
}

}  // namespace row_issue
