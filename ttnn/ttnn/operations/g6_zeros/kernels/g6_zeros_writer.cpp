// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros writer: the op's only kernel (no reader, no compute).
//
// Raw dataflow API, no kernel_lib helper:
//  - make_zero_source: dataflow_kernel_lib::zero_tile zeroes exactly one tile, but the source is
//    zero_chunk_bytes (16384 B is not a whole number of bfloat8_b tiles); it is the same
//    async_write_zeros + write_zeros_l1_barrier pair at a different size.
//  - write_zero_run: Noc::async_write_zeros overload 2 writes within one page per call and cannot
//    express a multi-page bank run; noc_async_write advances its source with its destination, so a
//    run would need a source as long as the run. A fixed zero_chunk_bytes source re-read per
//    single-packet write keeps the L1 footprint independent of the block extent.
//
// Work unit: a bank-major page range [range_start, range_start + range_count). Bank-major position
// q maps to (bank b, row r) with page id r * B + b; within one bank consecutive rows are
// consecutive addresses, so each bank's slice of the range is one contiguous DRAM run.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

struct BankRun {
    uint32_t bank;
    uint32_t first_row;
    uint32_t num_rows;
};

// Bank b holds rows_per_bank + (b < banks_with_extra_row) rows; its bank-major positions start at
// b * rows_per_bank + min(b, banks_with_extra_row).
FORCE_INLINE BankRun locate_run(uint32_t q, uint32_t range_end, uint32_t rows_per_bank, uint32_t banks_with_extra_row) {
    const uint32_t long_bank_rows = rows_per_bank + 1;
    const uint32_t long_banks_span = banks_with_extra_row * long_bank_rows;
    BankRun run;
    uint32_t rows_in_bank;
    if (q < long_banks_span) {
        run.bank = q / long_bank_rows;
        run.first_row = q - run.bank * long_bank_rows;
        rows_in_bank = long_bank_rows;
    } else {
        // Reached only when rows_per_bank > 0: with fewer pages than banks every page is in a long bank.
        const uint32_t q_short = q - long_banks_span;
        const uint32_t short_bank = q_short / rows_per_bank;
        run.bank = banks_with_extra_row + short_bank;
        run.first_row = q_short - short_bank * rows_per_bank;
        rows_in_bank = rows_per_bank;
    }
    const uint32_t rows_left_in_bank = rows_in_bank - run.first_row;
    const uint32_t rows_left_in_range = range_end - q;
    run.num_rows = rows_left_in_range < rows_left_in_bank ? rows_left_in_range : rows_left_in_bank;
    return run;
}

// One contiguous DRAM byte range, written as single-packet transfers that all read the same
// zero_chunk_bytes source. A chunk may end mid-page: the next chunk continues the same range.
template <uint32_t zero_chunk_bytes>
FORCE_INLINE void write_zero_run(uint32_t zero_source_addr, uint64_t run_noc_addr, uint32_t run_bytes) {
    while (run_bytes > zero_chunk_bytes) {
        noc_async_write_one_packet(zero_source_addr, run_noc_addr, zero_chunk_bytes);
        run_noc_addr += zero_chunk_bytes;
        run_bytes -= zero_chunk_bytes;
    }
    noc_async_write_one_packet(zero_source_addr, run_noc_addr, run_bytes);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_zero_source_id = get_compile_time_arg_val(0);
    constexpr uint32_t zero_chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t dram_bank_count = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    static_assert(zero_chunk_bytes <= NOC_MAX_BURST_SIZE, "a single-packet write cannot exceed the burst size");
    static_assert(zero_chunk_bytes % NOC_L1_READ_ALIGNMENT_BYTES == 0, "L1 zeroing is a NoC loopback read");
    static_assert(zero_chunk_bytes % NOC_DRAM_WRITE_ALIGNMENT_BYTES == 0, "every chunk start must stay aligned");
    static_assert(page_bytes % NOC_DRAM_WRITE_ALIGNMENT_BYTES == 0, "every run start must stay aligned");
    static_assert(dram_bank_count == NUM_DRAM_BANKS, "host bank count must match the interleave modulus");

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t range_start = get_arg_val<uint32_t>(0);
    const uint32_t range_count = get_arg_val<uint32_t>(1);
    if (range_count == 0) {
        return;
    }
    const uint32_t range_end = range_start + range_count;

    const uint32_t rows_per_bank = num_pages / dram_bank_count;
    const uint32_t banks_with_extra_row = num_pages - rows_per_bank * dram_bank_count;

    // page_bytes is the interleaved aligned page size, which makes a bank's consecutive rows
    // consecutive addresses.
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);

    Noc noc;
    CircularBuffer cb_zero_source(cb_zero_source_id);
    // Fresh CB with no reserve/push: read pointer == write pointer, so the zeroed bytes are the
    // write source.
    const uint32_t zero_source_addr = cb_zero_source.get_write_ptr();

    uint32_t q = range_start;
    BankRun run;
    uint64_t run_noc_addr;
    {
        DeviceZoneScopedN("make_zero_source");
        noc.async_write_zeros(cb_zero_source, zero_chunk_bytes);
        // The first run's geometry does not touch the source, so it fills the zeroing barrier's shadow.
        run = locate_run(q, range_end, rows_per_bank, banks_with_extra_row);
        run_noc_addr = output.get_noc_addr(run.first_row * dram_bank_count + run.bank);
        noc.write_zeros_l1_barrier();
    }

    {
        DeviceZoneScopedN("write_zero_runs");
        while (true) {
            write_zero_run<zero_chunk_bytes>(zero_source_addr, run_noc_addr, run.num_rows * page_bytes);
            q += run.num_rows;
            if (q >= range_end) {
                break;
            }
            run = locate_run(q, range_end, rows_per_bank, banks_with_extra_row);
            run_noc_addr = output.get_noc_addr(run.first_row * dram_bank_count + run.bank);
        }
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
