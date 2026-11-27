// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// page_copy and page_copy_row_major reader (NCRISC, NOC0). Each output page of this core's range is
// one input page (a tile, or a stick for ROW_MAJOR), fetched into cb_output_pages a block of
// block_pages at a time.
//
// No kernel_lib helper covers this step: the helpers that copy tiles are compute-side and round-trip
// values through DEST, which this op must not do (op_design.md, "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_concat_input_placement.hpp"
#include "g6_concat_page_sweep.hpp"
#include "g6_concat_tile_rows.hpp"

namespace {

// Common runtime args; layout mirrors _reader_common_args() / PAGE_READER_COMMON_FIRST_INPUT in
// g6_concat_program_descriptor.py.
constexpr uint32_t CRTA_BLOCK_PAGES = 0;
constexpr uint32_t CRTA_OUT_WT = 1;
constexpr uint32_t CRTA_OUT_HT = 2;
constexpr uint32_t CRTA_LAST_ROW_VALID_ROWS = 3;
constexpr uint32_t CRTA_NUM_INPUTS = 4;
constexpr uint32_t CRTA_FIRST_INPUT = 5;
constexpr uint32_t CRTA_WORDS_PER_INPUT = 2;  // base address, pages of the input per outer slice

FORCE_INLINE uint32_t input_base_address(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT);
}

FORCE_INLINE uint32_t input_pages_per_slice(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT + 1);
}

// Reads run_pages consecutive pages of one input into consecutive slots, in sweep_run order.
template <
    uint32_t page_bytes,
    uint32_t transfer_bytes,
    uint32_t face_row_bytes,
    uint32_t trim_rows_max,
    bool bank_stagger,
    typename Accessor>
FORCE_INLINE void fetch_run(
    const Accessor& input,
    uint32_t input_page,
    uint32_t run_pages,
    uint32_t stagger,
    uint32_t slot_addr,
    g6_concat::OutputRowCursor& rows) {
    g6_concat::sweep_run<page_bytes, transfer_bytes, trim_rows_max, bank_stagger>(
        run_pages,
        stagger,
        rows,
        [&](uint32_t j, uint32_t offset_bytes, uint32_t size_bytes) {
            const uint32_t dst = slot_addr + j * page_bytes;
            if (size_bytes == page_bytes) {
                noc_async_read_page(input_page + j, input, dst);
            } else {
                noc_async_read(input.get_noc_addr(input_page + j, offset_bytes), dst + offset_bytes, size_bytes);
            }
        },
        [&](uint32_t j, uint32_t valid_rows) {
            // Source and destination offsets are identical, so the read stays alignment-legal.
            const uint32_t dst = slot_addr + j * page_bytes;
            g6_concat::for_each_valid_row_piece<face_row_bytes>(
                valid_rows, [&](uint32_t offset_bytes, uint32_t size_bytes) {
                    noc_async_read(input.get_noc_addr(input_page + j, offset_bytes), dst + offset_bytes, size_bytes);
                });
        });
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(3);
    constexpr uint32_t transfer_bytes = get_compile_time_arg_val(4);
    constexpr bool bank_stagger = get_compile_time_arg_val(5) != 0;
    // 0: one block in flight, retired by a full read barrier. T >= 2: a ring of T blocks tagged
    // with transaction ids 1..T, the oldest retired by its own barrier (needs depth >= T).
    constexpr uint32_t read_trids = get_compile_time_arg_val(6);
    constexpr uint32_t depth = get_compile_time_arg_val(7);
    constexpr uint32_t input_l1_mask_lo = get_compile_time_arg_val(8);
    constexpr uint32_t input_l1_mask_hi = get_compile_time_arg_val(9);
    using DramInputArgs = TensorAccessorArgs<10>;
    using L1InputArgs = TensorAccessorArgs<DramInputArgs::next_compile_time_args_offset()>;
    using Inputs = g6_concat::InputPlacement<input_l1_mask_lo, input_l1_mask_hi, DramInputArgs, L1InputArgs>;
    static_assert(read_trids != 1 && read_trids <= depth, "a trid ring needs 2..depth blocks");

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    // Walk cursor (o, k, c') of start_page: output page o * C + Coff_k + c' is input k's page o * C_k + c'.
    uint32_t slice = get_arg_val<uint32_t>(2);
    uint32_t input_idx = get_arg_val<uint32_t>(3);
    uint32_t page_in_slice = get_arg_val<uint32_t>(4);
    const uint32_t stagger = get_arg_val<uint32_t>(5);

    const uint32_t block_pages = get_common_arg_val<uint32_t>(CRTA_BLOCK_PAGES);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(CRTA_NUM_INPUTS);
    g6_concat::OutputRowCursor rows(
        start_page,
        get_common_arg_val<uint32_t>(CRTA_OUT_WT),
        get_common_arg_val<uint32_t>(CRTA_OUT_HT),
        get_common_arg_val<uint32_t>(CRTA_LAST_ROW_VALID_ROWS));

    uint32_t base_address = input_base_address(input_idx);
    uint32_t pages_per_slice = input_pages_per_slice(input_idx);
    uint32_t input_page = slice * pages_per_slice + page_in_slice;

    // page_fetch_block: runs of one input, so the accessor is picked per run, not per page.
    auto fetch_block = [&](uint32_t block_extent, uint32_t slot_addr) {
        for (uint32_t page = 0; page < block_extent;) {
            const uint32_t left_in_input = pages_per_slice - page_in_slice;
            const uint32_t run_pages = block_extent - page < left_in_input ? block_extent - page : left_in_input;
            Inputs::with_accessor(input_idx, base_address, page_bytes, [&](const auto& input) {
                fetch_run<page_bytes, transfer_bytes, face_row_bytes, trim_rows_max, bank_stagger>(
                    input, input_page, run_pages, stagger, slot_addr, rows);
            });
            slot_addr += run_pages * page_bytes;
            page += run_pages;
            input_page += run_pages;
            page_in_slice += run_pages;
            if (page_in_slice == pages_per_slice) {
                page_in_slice = 0;
                if (++input_idx == num_inputs) {
                    input_idx = 0;
                    ++slice;
                }
                base_address = input_base_address(input_idx);
                pages_per_slice = input_pages_per_slice(input_idx);
                input_page = slice * pages_per_slice;
            }
        }
    };

    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    auto block_extent_of = [&](uint32_t block_idx) {
        return block_idx + 1 == num_blocks_this_core ? core_pages - block_idx * block_pages : block_pages;
    };

    if constexpr (read_trids == 0) {
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            {
                DeviceZoneScopedN("G6C-RESERVE");
                cb_reserve_back(cb_output_pages, block_pages);
            }
            {
                DeviceZoneScopedN("G6C-FETCH");
                fetch_block(block_extent_of(block_idx), get_write_ptr(cb_output_pages));
                noc_async_read_barrier();
            }
            // Nominal count on the ragged last block keeps the ring cycle closed.
            cb_push_back(cb_output_pages, block_pages);
        }
    } else {
        // Block k carries trid (k mod T) + 1, so the barrier on the id block k is about to reuse
        // retires block k - T. The CB write pointer only moves on push, so it lags the unpushed
        // blocks: block k lands in_flight blocks past it, wrapped at the end of the CB. The CB holds
        // depth whole blocks, so a block never straddles the wrap.
        Noc noc;
        const uint32_t block_bytes = block_pages * page_bytes;
        const uint32_t cb_base = get_write_ptr(cb_output_pages);
        const uint32_t cb_end = cb_base + depth * block_bytes;
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            const uint32_t trid = block_idx % read_trids + 1;
            if (block_idx >= read_trids) {
                DeviceZoneScopedN("G6C-RETIRE");
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
                cb_push_back(cb_output_pages, block_pages);
            }
            const uint32_t in_flight = block_idx < read_trids - 1 ? block_idx : read_trids - 1;
            {
                DeviceZoneScopedN("G6C-RESERVE");
                cb_reserve_back(cb_output_pages, (in_flight + 1) * block_pages);
            }
            uint32_t slot_addr = get_write_ptr(cb_output_pages) + in_flight * block_bytes;
            if (slot_addr >= cb_end) {
                slot_addr -= depth * block_bytes;
            }
            {
                DeviceZoneScopedN("G6C-FETCH");
                dataflow_kernel_lib::set_read_trid(noc, trid);
                fetch_block(block_extent_of(block_idx), slot_addr);
            }
        }
        const uint32_t first_unretired = num_blocks_this_core > read_trids ? num_blocks_this_core - read_trids : 0;
        for (uint32_t block_idx = first_unretired; block_idx < num_blocks_this_core; ++block_idx) {
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, block_idx % read_trids + 1);
            cb_push_back(cb_output_pages, block_pages);
        }
        dataflow_kernel_lib::set_read_trid(noc, 0);
    }
}
