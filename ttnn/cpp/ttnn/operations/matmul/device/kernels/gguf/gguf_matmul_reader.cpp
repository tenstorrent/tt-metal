// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight reader of the GGUF matmul: this core's records in compute order into c_0 as the decoder's L1 pages
// (experimental/gguf_page.h). One-page groups store codes then scale rows, the page's own order: each lands in its page
// with one read, a unit of them per transaction id, one or two units in flight. Larger groups land whole in a staging
// ring (c_2), one read each, one or two in flight, and go into their pages from local L1, each codes part with the
// group's scale rows. The scale rows are converted once, in the first page (prepare_scales), and the block is copied to
// the other pages. Codebook formats then fill in the pages' grid levels (lookup_grid) in GGUF_LOOKUP_PARTS parts of
// their index words: this kernel the first GGUF_READER_PARTS, the writer of this core the next GGUF_WRITER_PARTS
// (gguf_matmul_writer.cpp) on a request per record, the compute kernel the rest. GGUF_EXPANDED: the pages hold the
// level codes already (gguf_host_packing.cpp), each K tile without its index words, and no lookups run. Compile args:
// format id, c_0 pages, one-page groups per unit, staging slots, the lookup request, answer and page address
// semaphores, reads in flight, records per DRAM page (gguf_records_per_page: page (g / R, n) holds K groups g of N tile
// n consecutively), bytes per c_0 page (the decoder's L1 page, or one record when c_0 packs records), then the weight's
// TensorAccessorArgs. Decode, one-page groups with packed c_0: a unit is the records of one DRAM page, at most
// one-page-groups-per-unit of them, read with one transaction, never streamed. Decode, one-page groups: once the writer
// of this core streams (answer semaphore nonzero), it reads every other unit on its own NoC, so that a core's reads use
// both NoC ports of their DRAM bank; this kernel reserves the unit's pages, grants it (page address semaphore: its
// first record, request semaphore: the count of grants) and pushes it when the writer has landed it (answer semaphore:
// 1 + units landed).
// Runtime args: weight address, N tiles, runs, then per run its N tile, first and end K group. With GGUF_PREFILL:
// weight address, N tiles, first N tile, N tiles of this core, K groups, M chunks, first K group, N tiles per pass.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/gguf_page.h"

alignas(4) std::uint16_t table_copy[1024 / 2];  // local_level_table

void kernel_main() {
    using namespace experimental::gguf;
    constexpr auto FMT = static_cast<Format>(get_compile_time_arg_val(0));
    // IQ2_S's 2 KiB level table is copied to the stack: a static copy would be zeroed at every launch. The 4 KiB tables
    // (IQ1_S, IQ1_M) stay in L1, as the stack has no room for them.
    alignas(4) std::uint16_t stack_copy[table_entries<FMT>() == 1024 ? 1024 : 2];
    constexpr uint32_t ring = get_compile_time_arg_val(1);
    constexpr uint32_t unit = get_compile_time_arg_val(2);
    constexpr uint32_t slots = get_compile_time_arg_val(3);
    constexpr uint32_t request_sem = get_compile_time_arg_val(4), answer_sem = get_compile_time_arg_val(5);
    constexpr uint32_t address_sem = get_compile_time_arg_val(6);
    constexpr uint32_t in_flight = get_compile_time_arg_val(7);
    constexpr uint32_t writer_parts = GGUF_WRITER_PARTS;
    constexpr uint32_t rpp = get_compile_time_arg_val(8);   // records per DRAM page
    constexpr uint32_t page = get_compile_time_arg_val(9);  // bytes per c_0 page
    constexpr auto weight_args = TensorAccessorArgs<10>();
    constexpr PageLayout L = page_layout<FMT>();
    constexpr uint32_t P = L.group_tiles / L.page_tiles;  // pages per group
    constexpr uint32_t scale_bytes = L.scale_rows * 64;
    // Expanded: per K tile the page's first `head` bytes (level codes, sign words), then `tail` bytes of the decoder's
    // that follow the index words (IQ1_S, IQ1_M, IQ2_S).
    constexpr bool expanded = GGUF_EXPANDED;
    constexpr uint32_t index_bytes = 4 * index_words<FMT>();
    constexpr uint32_t head = index_offset<FMT>();
    constexpr uint32_t tail = expanded ? expanded_tail<FMT>() : 0;
    constexpr uint32_t tile_bytes = expanded ? head + tail : L.code_bytes;
    constexpr uint32_t record = scale_bytes + P * tile_bytes;
    constexpr bool packed = page == record && rpp > 1;  // c_0 pages hold records back to back

    const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(0));
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    struct Rec {
        uint32_t g, n;  // K group, N tile
    };
    auto page_of = [&](Rec r) { return r.g / rpp * n_tiles + r.n; };
    constexpr auto offset_of = [](Rec r) { return r.g % rpp * record; };
#ifdef GGUF_PREFILL
    // Prefill (gguf_matmul_prefill_program_factory.cpp): N tiles [n0, n0 + nc) in passes of up to np; per pass, per M
    // chunk, per K group, the pass's records.
    const uint32_t n0 = get_arg_val<uint32_t>(2), nc = get_arg_val<uint32_t>(3);
    const uint32_t groups = get_arg_val<uint32_t>(4), mchunks = get_arg_val<uint32_t>(5);
    const uint32_t g0 = get_arg_val<uint32_t>(6), np = get_arg_val<uint32_t>(7);
    const uint32_t total = mchunks * groups * nc;
    uint32_t first = n0, width = nc < np ? nc : np, chunk = 0, group = 0, column = 0;  // first: the pass's first N tile
    auto next_record = [&]() {
        const Rec id{g0 + group, first + column};
        if (++column == width) {
            column = 0;
            if (++group == groups) {
                group = 0;
                if (++chunk == mchunks) {
                    chunk = 0;
                    first += np;
                    width = n0 + nc - first < np ? n0 + nc - first : np;
                }
            }
        }
        return id;
    };
#else
    const uint32_t runs = get_arg_val<uint32_t>(2);
    uint32_t total = 0;
    for (uint32_t r = 0; r < runs; r++) {
        total += get_arg_val<uint32_t>(5 + 3 * r) - get_arg_val<uint32_t>(4 + 3 * r);
    }
    // The next record to read: records are read in order.
    uint32_t run = 0, group = get_arg_val<uint32_t>(4);
    // Records left in the current run's DRAM page, from the next one on.
    auto page_left = [&]() {
        const uint32_t end = get_arg_val<uint32_t>(5 + 3 * run), page_end = (group / rpp + 1) * rpp;
        return (end < page_end ? end : page_end) - group;
    };
    auto next_record = [&]() {
        const Rec id{group, get_arg_val<uint32_t>(3 + 3 * run)};
        if (++group == get_arg_val<uint32_t>(5 + 3 * run) && ++run < runs) {
            group = get_arg_val<uint32_t>(4 + 3 * run);
        }
        return id;
    };
#endif

    Noc noc;
    CircularBuffer pages(tt::CBIndex::c_0);
    static_assert(P > 1 || (L.code_offset == 0 && L.fp16_rows == 0 && L.scale_offset == L.code_bytes));
    if constexpr (P == 1) {
        // idx pages past the write pointer, in the ring.
        uint32_t pushed = 0;
        auto offset = [&](uint32_t idx) { return (((pushed + idx) % ring) - (pushed % ring)) * page; };
        uint32_t issued = 0, pending = 0, units_issued = 0, units_done = 0, sizes[in_flight];
#ifdef GGUF_PREFILL
        constexpr bool stream = false;
#else
        constexpr bool stream = !packed;  // the factory does not stream packed weights
#endif
        Semaphore grants(request_sem), first(address_sem), landed(answer_sem);
        uint32_t granted = 0;
        bool by_writer[in_flight];
        auto issue = [&]() {
            uint32_t n = total - issued < unit ? total - issued : unit;
#ifndef GGUF_PREFILL
            if constexpr (packed) {
                // One DRAM page's records, not across the end of the ring.
                const uint32_t left = page_left(), to_end = ring - (pushed + pending) % ring;
                n = n < left ? n : left;
                n = n < to_end ? n : to_end;
            }
#endif
            pages.reserve_back(pending + n);
            const bool to_writer = stream && units_issued % 2 == 1 && landed.value() > 0;
            by_writer[units_issued % in_flight] = to_writer;
            for (uint32_t j = 0; j < n; j++) {
                const Rec r = next_record();
                if (!to_writer && (!packed || j == 0)) {
                    noc.async_read<NocOptions::TXN_ID>(
                        weights,
                        pages,
                        packed ? n * record : record,
                        {.page_id = page_of(r), .offset_bytes = offset_of(r)},
                        {.offset_bytes = offset(pending + j)},
                        {.trid = 1 + units_issued % in_flight});
                }
            }
            if (to_writer) {
                first.set(issued);
                asm volatile("fence" ::: "memory");  // the first record before the grant
                grants.set(++granted);
            }
            sizes[units_issued % in_flight] = n;
            pending += n;
            issued += n;
            units_issued++;
        };
        while (units_issued < in_flight && issued < total) {
            issue();
        }
        uint32_t writer_units = 0;
        while (units_done < units_issued) {
            if (by_writer[units_done % in_flight]) {
                landed.wait_min(1 + ++writer_units);
            } else {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + units_done % in_flight});
            }
            const uint32_t n = sizes[units_done % in_flight];
            pages.push_back(n);
            pushed += n;
            pending -= n;
            units_done++;
            if (issued < total) {
                issue();
            }
        }
    } else {
        static_assert(slots > in_flight);
        constexpr uint32_t local = slots + 1;  // transaction id of the copies within L1
        constexpr bool shared_scales = L.fp16_rows > 0;
        CircularBuffer stage(tt::CBIndex::c_2);
        stage.reserve_back(1);  // private: never pushed
        const uint32_t stage_addr = stage.get_write_ptr();
        const UnicastEndpoint self;
        const uint32_t me_x = my_x[noc.get_noc_id()], me_y = my_y[noc.get_noc_id()];
        auto read_record = [&](uint32_t i) {
            noc.async_read<NocOptions::TXN_ID>(
                weights,
                stage,
                record,
                {.page_id = page_of(next_record())},
                {.offset_bytes = (i % slots) * record},
                {.trid = 1 + i % slots});
        };
        auto copy = [&](uint32_t from, uint32_t bytes, uint32_t to) {
            noc.async_read<NocOptions::TXN_ID>(
                self,
                pages,
                bytes,
                {.noc_x = me_x, .noc_y = me_y, .addr = from},
                {.offset_bytes = to},
                {.trid = local});
        };
        for (uint32_t i = 0; i < in_flight && i < total; i++) {
            read_record(i);
        }
        [[maybe_unused]] const std::uint16_t* table = nullptr;
        Semaphore requests(request_sem), answers(answer_sem), address(address_sem);
        uint32_t asked = 0;
        if constexpr (is_codebook<FMT>() && !expanded) {
            if constexpr (table_entries<FMT>() * 2 <= 1024) {
                table = local_level_table<FMT>(table_copy);
            } else if (total >= 3) {  // the copy pays off over a few records
                table = local_level_table<FMT, sizeof(stack_copy)>(stack_copy);
            } else {
                table = level_table<FMT>();
            }
        }
        for (uint32_t i = 0; i < total; i++) {
            if (i + in_flight < total) {
                read_record(i + in_flight);  // its slot held record i - 1, whose copies are done
            }
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + i % slots});
            pages.reserve_back(P);
            const uint32_t from = stage_addr + (i % slots) * record;
            for (uint32_t p = 0; p < P; p++) {
                if constexpr (expanded) {
                    copy(from + scale_bytes + p * tile_bytes, head, p * page);
                    if constexpr (tail > 0) {
                        copy(from + scale_bytes + p * tile_bytes + head, tail, p * page + head + index_bytes);
                    }
                } else {
                    copy(from + scale_bytes + p * L.code_bytes, L.code_bytes, p * page + L.code_offset);
                }
                if (p == 0 || !shared_scales) {
                    copy(from, scale_bytes, p * page + L.scale_offset);
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = local});
            invalidate_l1_cache();
            const uint32_t first = pages.get_write_ptr();
            // A writer with the larger share of the lookups gets the record before the scale work, which it then
            // overlaps.
            [[maybe_unused]] constexpr bool early = writer_parts > GGUF_READER_PARTS;
            [[maybe_unused]] auto request = [&]() {
                address.set(first);
                asm volatile("fence" ::: "memory");  // the address is in L1 before the request
                requests.set(++asked);
            };
            if constexpr (is_codebook<FMT>() && !expanded && writer_parts > 0 && early) {
                request();
            }
            if constexpr (shared_scales) {
                prepare_scales<FMT>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + L.scale_offset));
                asm volatile("fence" ::: "memory");
                for (uint32_t p = 1; p < P; p++) {
                    copy(first + L.scale_offset, L.scale_bytes, p * page + L.scale_offset);
                }
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = local});
            }
            if constexpr (is_codebook<FMT>() && !expanded) {
                if constexpr (writer_parts > 0 && !early) {
                    request();
                }
                for (uint32_t p = 0; p < P; p++) {
                    if constexpr (GGUF_READER_PARTS > 0) {
                        lookup_grid<FMT>(
                            reinterpret_cast<const uint32_t*>(first + p * page + index_offset<FMT>()),
                            table,
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + p * page),
                            0,
                            GGUF_LOOKUP_PARTS,
                            GGUF_READER_PARTS);
                    }
                }
                asm volatile("fence" ::: "memory");
                if constexpr (writer_parts > 0) {
                    answers.wait_min(asked);
                }
            }
            pages.push_back(P);
        }
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
