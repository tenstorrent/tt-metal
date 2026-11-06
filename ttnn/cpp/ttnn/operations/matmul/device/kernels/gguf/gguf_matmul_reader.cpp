// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight reader of the GGUF matmul: this core's records in compute order into c_0 as the decoder's L1 pages
// (experimental/gguf_page.h). One-page groups store codes then scale rows, the page's own order: each lands in its
// page with one read, a unit of them per transaction id, two units in flight. Larger groups land whole in a staging
// ring (c_2), one read each, two in flight, and go into their pages from local L1, each codes part with the group's
// scale rows. Codebook formats then fill in each page's grid levels and fp32 scale copies (prepare_page); the others
// convert the scale rows once, in the first page (prepare_scales), and copy the block to the other pages.
// Compile args: format id, c_0 pages, one-page groups per unit, staging slots, then the weight's TensorAccessorArgs.
// Runtime args: weight address, N tiles, runs, then per run its N tile, first and end K group.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/gguf_page.h"

void kernel_main() {
    using namespace experimental::gguf;
    constexpr auto FMT = static_cast<Format>(get_compile_time_arg_val(0));
    constexpr uint32_t ring = get_compile_time_arg_val(1);
    constexpr uint32_t unit = get_compile_time_arg_val(2);
    constexpr uint32_t slots = get_compile_time_arg_val(3);
    constexpr auto weight_args = TensorAccessorArgs<4>();
    constexpr PageLayout L = page_layout<FMT>();
    constexpr uint32_t P = L.group_tiles / L.page_tiles;  // pages per group
    constexpr uint32_t page = L.rows * 64;
    constexpr uint32_t scale_bytes = L.scale_rows * 64;
    constexpr uint32_t record = scale_bytes + P * L.code_bytes;
    constexpr uint32_t in_flight = 2;

    const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(0));
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t runs = get_arg_val<uint32_t>(2);
    uint32_t total = 0;
    for (uint32_t r = 0; r < runs; r++) {
        total += get_arg_val<uint32_t>(5 + 3 * r) - get_arg_val<uint32_t>(4 + 3 * r);
    }
    // The page id of the next record to read: records are read in order.
    uint32_t run = 0, group = get_arg_val<uint32_t>(4);
    auto next_record = [&]() {
        const uint32_t id = group * n_tiles + get_arg_val<uint32_t>(3 + 3 * run);
        if (++group == get_arg_val<uint32_t>(5 + 3 * run) && ++run < runs) {
            group = get_arg_val<uint32_t>(4 + 3 * run);
        }
        return id;
    };

    Noc noc;
    CircularBuffer pages(tt::CBIndex::c_0);
    static_assert(P > 1 || (L.code_offset == 0 && L.fp16_rows == 0 && L.scale_offset == L.code_bytes));
    if constexpr (P == 1) {
        // idx pages past the write pointer, in the ring.
        uint32_t pushed = 0;
        auto offset = [&](uint32_t idx) { return (((pushed + idx) % ring) - (pushed % ring)) * page; };
        uint32_t issued = 0, pending = 0, units_issued = 0, units_done = 0, sizes[in_flight];
        auto issue = [&]() {
            const uint32_t n = total - issued < unit ? total - issued : unit;
            pages.reserve_back(pending + n);
            for (uint32_t j = 0; j < n; j++) {
                noc.async_read<NocOptions::TXN_ID>(
                    weights,
                    pages,
                    record,
                    {.page_id = next_record()},
                    {.offset_bytes = offset(pending + j)},
                    {.trid = 1 + units_issued % in_flight});
            }
            sizes[units_issued % in_flight] = n;
            pending += n;
            issued += n;
            units_issued++;
        };
        while (units_issued < in_flight && issued < total) {
            issue();
        }
        while (units_done < units_issued) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + units_done % in_flight});
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
        constexpr bool shared_scales = L.fp16_rows > 0 && !is_codebook<FMT>();
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
                {.page_id = next_record()},
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
        for (uint32_t i = 0; i < total; i++) {
            if (i + in_flight < total) {
                read_record(i + in_flight);  // its slot held record i - 1, whose copies are done
            }
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + i % slots});
            pages.reserve_back(P);
            const uint32_t from = stage_addr + (i % slots) * record;
            for (uint32_t p = 0; p < P; p++) {
                copy(from + scale_bytes + p * L.code_bytes, L.code_bytes, p * page + L.code_offset);
                if (p == 0 || !shared_scales) {
                    copy(from, scale_bytes, p * page + L.scale_offset);
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = local});
            invalidate_l1_cache();
            const uint32_t first = pages.get_write_ptr();
            if constexpr (shared_scales) {
                prepare_scales<FMT>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + L.scale_offset));
                asm volatile("fence" ::: "memory");
                for (uint32_t p = 1; p < P; p++) {
                    copy(first + L.scale_offset, L.scale_bytes, p * page + L.scale_offset);
                }
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = local});
            } else {
                for (uint32_t p = 0; p < P; p++) {
                    prepare_page<FMT>(first + p * page);
                }
                asm volatile("fence" ::: "memory");
            }
            pages.push_back(P);
        }
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
