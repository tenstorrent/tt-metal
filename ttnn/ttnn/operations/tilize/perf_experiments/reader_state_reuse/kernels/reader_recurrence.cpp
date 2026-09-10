// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — variant 3/5: RECURRENCE.
//
// RAW-LLK JUSTIFICATION (bypasses `TensorAccessor::get_noc_addr()` /
// `InterleavedAddrGen::get_noc_addr` — dataflow_api_addrgen.h:302-311, the
// same code `reader_raw.cpp` and the real op's `reader_read_block` both call
// through unmodified): that path resolves EVERY page id independently —
// `get_bank_offset_index<true>(id)` is `udivsi3_const_divisor<NUM_DRAM_BANKS>`
// on this box (12 banks, `IS_NOT_POW2_NUM_DRAM_BANKS` — dataflow_api_addrgen.h
// :19-25), i.e. a software constant-divisor division PER STICK. But each
// reserve/push chunk's 32 stick ids (`chunk_start_page .. +31`) are
// CONSECUTIVE, so `bank_offset_index`/`bank_index` form a recurrence with NO
// division in it: the bank increments by 1 and wraps every `NUM_DRAM_BANKS`,
// rolling `bank_offset_index` over on wrap. This file computes that
// recurrence directly against the same globals `InterleavedAddrGen` itself
// reads (`interleaved_addr_gen::get_bank_offset`, `get_noc_xy`,
// `get_noc_addr_helper` — dataflow_api_addrgen.h:44-60,222-228), so the
// address is bit-identical to the accessor's, just reached without a division
// after each chunk's FIRST page (one division per 32-stick chunk instead of
// one per stick). `noc_async_read` itself (the NoC issue mechanism) is
// untouched — this variant isolates the address-recurrence idea alone, before
// combining it with a state-reused NoC command (`reader_state_grouped.cpp`).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t kTileH = 32;
}

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t bw = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();
    constexpr uint32_t num_chunks = rows / kTileH;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t byte_offset = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr);
    const uint32_t aligned_page_size = in_acc.get_aligned_page_size();
    const uint32_t bank_base = in_acc.get_bank_base_address();

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t chunk_start_page = start_page + chunk * kTileH;
        // ONE division per chunk: chunk_start_page -> (bank_offset_index, bank).
        uint32_t boi = interleaved_addr_gen::get_bank_offset_index<true>(chunk_start_page);
        uint32_t bank = interleaved_addr_gen::get_bank_index<true>(chunk_start_page, boi);

        uint32_t l1_addr;
        {
            MaybeDeviceZoneScope("reader_reserve");
            cb_reserve_back(cb_in, bw);
            l1_addr = get_write_ptr(cb_in);
        }
        {
            MaybeDeviceZoneScope("reader_issue");
            uint32_t addr = l1_addr;
            for (uint32_t row = 0; row < kTileH; ++row) {
                const uint32_t dram_addr = boi * aligned_page_size + bank_base + byte_offset +
                                           interleaved_addr_gen::get_bank_offset<true>(bank);
                const uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank, noc_index);
                const uint64_t noc_addr = get_noc_addr_helper(noc_xy, dram_addr);
                noc_async_read(noc_addr, addr, row_bytes);
                addr += row_bytes;
                // recurrence: consecutive page ids step the bank by 1, wrapping
                // every NUM_DRAM_BANKS (rolling bank_offset_index) -- no
                // division, no modulo.
                ++bank;
                if (bank == NUM_DRAM_BANKS) {
                    bank = 0;
                    ++boi;
                }
            }
        }
        {
            MaybeDeviceZoneScope("reader_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_in, bw);
    }
}
