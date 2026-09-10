// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — variant 5/5: STATE_GROUPED (the real candidate:
// recurrence addressing + bank-grouped NoC state reuse, idea 3+4 combined).
//
// RAW-LLK JUSTIFICATION. Bypasses two things `read_sticks_for_tilize` /
// `TensorAccessor::get_noc_addr()` cannot express at all:
//   (a) the per-page division (see reader_recurrence.cpp) -- closed the same
//       way, by a bank/bank_offset_index recurrence over consecutive page ids;
//   (b) `noc_async_read()`'s full per-transaction command programming
//       (NOC_TARG_ADDR_COORDINATE + NOC_TARG_ADDR_LO + NOC_RET_ADDR_LO +
//       NOC_AT_LEN_BE + NOC_CMD_CTRL, every call -- ncrisc_noc_fast_read,
//       noc_nonblocking_api.h:415-438) -- closed by reordering EACH 32-stick
//       reserve/push chunk's reads into BANK GROUPS and using
//       `noc_async_read_one_packet_set_state` ONCE per group (programs the
//       group's fixed NOC_TARG_ADDR_COORDINATE + the chunk-wide-constant
//       NOC_AT_LEN_BE = row_bytes) followed by
//       `noc_async_read_one_packet_with_state` per stick in that group
//       (programs only NOC_TARG_ADDR_LO + NOC_RET_ADDR_LO + NOC_CMD_CTRL).
//
// WHY GROUPING IS REQUIRED (reader_state_naive.cpp's null result): the DRAM
// target's NoC (x,y) is a function of the BANK, and consecutive stick page
// ids almost never share a bank on this box (12 banks, so bank cycles nearly
// every read) -- `with_state` cannot skip a coordinate write it would still
// need. Grouping same-bank reads together first makes the coordinate genuinely
// constant across a run of `with_state` calls, which is the only regime where
// state reuse pays for anything.
//
// CORRECTNESS OF THE REORDER: a chunk's 32 stick reads are mutually
// independent (each writes its own fixed offset `row * row_bytes` inside the
// ONE `cb_reserve_back`'d chunk region) and all land behind a SINGLE shared
// barrier before the chunk's CB pages are pushed -- exactly like the helper
// and every other variant here. Issuing them in bank order instead of row
// order changes nothing about which bytes land where.
//
// Within one bank, consecutive occurrences (rows `first, first+NUM_DRAM_BANKS,
// first+2*NUM_DRAM_BANKS, ...`) are themselves a recurrence: bank_offset_index
// increments by exactly 1 per occurrence, so the low target address just
// advances by `aligned_page_size` -- no division, no multiply, per occurrence.

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
        const uint32_t start_boi = interleaved_addr_gen::get_bank_offset_index<true>(chunk_start_page);
        const uint32_t start_bank = interleaved_addr_gen::get_bank_index<true>(chunk_start_page, start_boi);

        uint32_t l1_addr;
        {
            MaybeDeviceZoneScope("reader_reserve");
            cb_reserve_back(cb_in, bw);
            l1_addr = get_write_ptr(cb_in);
        }
        {
            MaybeDeviceZoneScope("reader_issue");
            // Bank groups within this 32-stick chunk: for each bank, the first
            // row assigned to it (a compare, not a division -- `first_row <
            // NUM_DRAM_BANKS` and `start_bank < NUM_DRAM_BANKS`, so their sum is
            // below `2 * NUM_DRAM_BANKS`) and then every NUM_DRAM_BANKS'th row
            // after it.
            for (uint32_t bank = 0; bank < NUM_DRAM_BANKS; ++bank) {
                const uint32_t first_row =
                    (bank >= start_bank) ? (bank - start_bank) : (bank - start_bank + NUM_DRAM_BANKS);
                if (first_row >= kTileH) {
                    continue;  // this bank isn't touched by this chunk at all
                }
                const uint32_t boi_first = start_boi + (((start_bank + first_row) >= NUM_DRAM_BANKS) ? 1u : 0u);
                const uint32_t bank_offset = interleaved_addr_gen::get_bank_offset<true>(bank);
                const uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank, noc_index);

                uint32_t dram_addr = boi_first * aligned_page_size + bank_base + byte_offset + bank_offset;
                const uint64_t first_noc_addr = get_noc_addr_helper(noc_xy, dram_addr);

                // ONE set_state per bank group: programs the coordinate
                // (constant for every read in this group) and the size
                // (constant for the whole chunk).
                noc_async_read_one_packet_set_state(first_noc_addr, row_bytes);
                noc_async_read_one_packet_with_state(dram_addr, l1_addr + first_row * row_bytes);

                for (uint32_t row = first_row + NUM_DRAM_BANKS; row < kTileH; row += NUM_DRAM_BANKS) {
                    dram_addr += aligned_page_size;  // recurrence: same bank, next bank_offset_index
                    noc_async_read_one_packet_with_state(dram_addr, l1_addr + row * row_bytes);
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
