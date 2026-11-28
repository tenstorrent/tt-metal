// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// One NoC transaction per interleaved ROW_MAJOR page for N consecutive pages: the byte segment
// [offset, offset + bytes) of each page, to / from an L1 stage with a fixed row stride.
//
// Helper bypass (measured, Perf 1): noc_async_write / noc_async_read with a
// TensorAccessor::get_noc_addr per row cost ~48 cycles per row (a divide by the bank count, two
// bank-table loads, the full 6-register command and two L1 counter updates); on small cells those
// rows are the critical chain. Here the page's bank and bank slot are stepped instead of divided,
// the command buffer is programmed once per call (*_one_packet_set_state) and per row only the
// bank coordinate register, the two local addresses and CMD_CTRL are written, with the issued /
// acked counters bumped once per call: ~28 cycles per row (c2025 store_block 1554 -> 886 cycles for
// 32 writes; c49 row reads 47.6 -> 30.0 cycles each). No dataflow API changes only the coordinate
// of a stateful command, so that register is written raw (NOC_CMD_BUF_WRITE_REG after the
// cmd_buf_ready poll) and the issue is ncrisc_noc_{write,read}_with_state<one_packet,
// skip_cmdbuf_chk>.
// A sharded accessor, a segment above one NoC packet or a single row takes the plain per-row path.
// The counters are bumped before any flush or barrier of the caller reads them.
// ReadPacer<gap, burst> holds a core's row reads to one per `gap` cycles in steady state, with up to
// `burst` reads let through at once (a token bucket on a virtual clock; gap 0 compiles it out). Full-
// grid row reads from L1 banks collapse under sustained issue at the fast rate (c38, 512-B rows: 28.6
// -> 37.3 us), while a cell with few rows per core is bound by their latency, so the host sets the gap
// for an L1 source and lets one tile-row through unpaced.

namespace g6_to_layout::row_issue {

template <uint32_t gap, uint32_t burst>
struct ReadPacer {
    uint32_t ready = get_timestamp_32b() - gap * burst;  // virtual issue time of the next read
    FORCE_INLINE void wait() {
        if constexpr (gap != 0) {
            const uint32_t now = get_timestamp_32b();
            // Idle time earns at most `burst` reads of credit.
            if (static_cast<int32_t>(now - gap * burst - ready) > 0) {
                ready = now - gap * burst;
            }
            ready += gap;
            while (static_cast<int32_t>(get_timestamp_32b() - ready) < 0) {
            }
        }
    }
};

// Consecutive interleaved pages: bank = page % NB walks the banks in order; the bank-local slot of
// the page advances by one aligned page each time the bank index wraps.
template <bool DRAM>
struct PageCursor {
    static constexpr uint32_t NB = DRAM ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    uint32_t bank;
    uint32_t slot;  // bank_base + (page / NB) * aligned_page + offset, without the bank's own offset
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

// Dynamic-NoC mode keeps its transaction counters elsewhere: they are bumped per row there.
constexpr bool per_row_counter = noc_mode == DM_DYNAMIC_NOC;

template <typename Accessor>
FORCE_INLINE void write_rows(
    const Accessor& out,
    uint32_t first_page,
    uint32_t rows,
    uint32_t offset,
    uint32_t bytes,
    uint32_t src,
    uint32_t src_stride) {
    if constexpr (!Accessor::DSpec::is_interleaved) {
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write(src, out.get_noc_addr(first_page + r, offset), bytes);
            src += src_stride;
        }
    } else {
        const uint8_t noc = noc_index;
        auto c = cursor_at(out, first_page, offset);
        if (bytes > NOC_MAX_BURST_SIZE || rows < 2) {
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_write(src, c.noc_addr(noc), bytes);
                src += src_stride;
                c.step();
            }
            return;
        }
        noc_async_write_one_packet_set_state(c.noc_addr(noc), bytes, noc);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t xy = c.xy(noc);
            const uint32_t dst = c.local();
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, xy);
            ncrisc_noc_write_with_state<noc_mode, false, per_row_counter, true, true>(noc, write_cmd_buf, src, dst);
            src += src_stride;
            c.step();
        }
        if constexpr (!per_row_counter) {
            noc_nonposted_writes_num_issued[noc] += rows;
            noc_nonposted_writes_acked[noc] += rows;
        }
    }
}

template <typename Accessor, uint32_t gap, uint32_t burst>
FORCE_INLINE void read_rows(
    const Accessor& in,
    uint32_t first_page,
    uint32_t rows,
    uint32_t offset,
    uint32_t bytes,
    uint32_t dst,
    uint32_t dst_stride,
    ReadPacer<gap, burst>& pacer) {
    if constexpr (!Accessor::DSpec::is_interleaved) {
        for (uint32_t r = 0; r < rows; ++r) {
            pacer.wait();
            noc_async_read(in.get_noc_addr(first_page + r, offset), dst, bytes);
            dst += dst_stride;
        }
    } else {
        const uint8_t noc = noc_index;
        auto c = cursor_at(in, first_page, offset);
        if (bytes > NOC_MAX_BURST_SIZE || rows < 2) {
            for (uint32_t r = 0; r < rows; ++r) {
                pacer.wait();
                noc_async_read(c.noc_addr(noc), dst, bytes);
                dst += dst_stride;
                c.step();
            }
            return;
        }
        noc_async_read_one_packet_set_state(c.noc_addr(noc), bytes, 0, noc);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t xy = c.xy(noc);
            const uint32_t src = c.local();
            pacer.wait();
            while (!noc_cmd_buf_ready(noc, read_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, xy);
            ncrisc_noc_read_with_state<noc_mode, per_row_counter, true, true>(noc, read_cmd_buf, src, dst);
            dst += dst_stride;
            c.step();
        }
        if constexpr (!per_row_counter) {
            noc_reads_num_issued[noc] += rows;
        }
    }
}

}  // namespace g6_to_layout::row_issue
