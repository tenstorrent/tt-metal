// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight reader of the GGUF matmul for TQ2_0: this core's records in compute order into c_0 (64-byte pages), one read
// each, a unit of them per transaction id, one or two units in flight. A record (gguf_host_packing.cpp group_tq2_0) is
// the eight BFP2_b weight tiles of one K group of one N tile, then the fp16 scale d of each of its 32 columns. Each
// record's scales go as fp32 into the scale tile of its window of K groups (c_3): group g's into row g % window.
// Compile args: c_0 records, records per unit, reads in flight, K groups per window, c_3 tiles, c_3 tile bytes, then
// the weight's TensorAccessorArgs. Runtime args: weight address, N tiles, runs, then per run its N tile, first and end
// K group.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/gguf_page.h"

void kernel_main() {
    constexpr uint32_t ring = get_compile_time_arg_val(0);
    constexpr uint32_t unit = get_compile_time_arg_val(1);
    constexpr uint32_t in_flight = get_compile_time_arg_val(2);
    constexpr uint32_t window = get_compile_time_arg_val(3);
    constexpr uint32_t scale_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t scale_tile_bytes = get_compile_time_arg_val(5);
    constexpr auto weight_args = TensorAccessorArgs<6>();
    constexpr uint32_t record = 2112, codes = 2048, record_pages = record / 64;

    const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(0));
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t runs = get_arg_val<uint32_t>(2);
    auto run_n = [](uint32_t r) { return get_arg_val<uint32_t>(3 + 3 * r); };
    auto run_g0 = [](uint32_t r) { return get_arg_val<uint32_t>(4 + 3 * r); };
    auto run_g1 = [](uint32_t r) { return get_arg_val<uint32_t>(5 + 3 * r); };
    uint32_t total = 0;
    for (uint32_t r = 0; r < runs; r++) {
        total += run_g1(r) - run_g0(r);
    }
    if (total == 0) {
        return;
    }

    Noc noc;
    CircularBuffer pages(tt::CBIndex::c_0);
    CircularBuffer scales(tt::CBIndex::c_3);
    // Scale tiles start zeroed: rows no record writes must be finite (they meet zero partials).
    noc.async_write_zeros(scales, scale_tiles * scale_tile_bytes);
    noc.write_zeros_l1_barrier();

    // Records are read in order: the page id of the next one to read.
    uint32_t run = 0, group = run_g0(0);
    auto next_record = [&]() {
        const uint32_t id = group * n_tiles + run_n(run);
        if (++group == run_g1(run) && ++run < runs) {
            group = run_g0(run);
        }
        return id;
    };
    // ... and the K group of the next one to hand to the compute kernel.
    uint32_t done_run = 0, done_group = run_g0(0);
    bool open = false;  // a scale tile is reserved

    uint32_t pushed = 0;  // records pushed
    // The c_0 offset of the record idx records past the write pointer, in the ring.
    auto offset = [&](uint32_t idx) { return (((pushed + idx) % ring) - (pushed % ring)) * record; };
    uint32_t issued = 0, pending = 0, units_issued = 0, units_done = 0, sizes[in_flight];
    auto issue = [&]() {
        const uint32_t n = total - issued < unit ? total - issued : unit;
        pages.reserve_back((pending + n) * record_pages);
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
        // The unit's records go to the compute kernel first; this kernel reads their scales before it issues the next
        // reads, so their slots are not refilled meanwhile.
        uint32_t at[unit];
        for (uint32_t j = 0; j < n; j++) {
            at[j] = pages.get_write_ptr() + offset(j) + codes;
        }
        pages.push_back(n * record_pages);
        pushed += n;
        pending -= n;
        units_done++;
        invalidate_l1_cache();
        for (uint32_t j = 0; j < n; j++) {
            const uint32_t g = done_group, row = g % window;
            bool last = false;  // of its window
            if (++done_group == run_g1(done_run)) {
                last = true;
                if (++done_run < runs) {
                    done_group = run_g0(done_run);
                }
            } else {
                last = done_group % window == 0;
            }
            if (!open) {
                scales.reserve_back(1);
                open = true;
            }
            // Columns 0..15 into face 0's row, 16..31 into face 1's (1 KiB further).
            const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at[j]);
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scales.get_write_ptr() + 64 * row);
            for (uint32_t w = 0; w < 16; w++) {
                const uint32_t pair = src[w];
                const uint32_t c = 2 * w + (w >= 8 ? 256 - 16 : 0);  // word of column 2w in the tile
                dst[c] = experimental::gguf::fp16_to_fp32(pair & 0xFFFF);
                dst[c + 1] = experimental::gguf::fp16_to_fp32(pair >> 16);
            }
            if (last) {
                scales.push_back(1);
                open = false;
            }
        }
        if (issued < total) {
            issue();
        }
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
