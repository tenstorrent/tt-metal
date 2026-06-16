// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(4);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t num_banks = get_compile_time_arg_val(7);
constexpr uint32_t enable_bank_packing = get_compile_time_arg_val(8);
constexpr uint32_t input_args_cta_idx = 9;
constexpr uint32_t input_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);                             // pages for this core
    uint32_t page_start_offset = get_arg_val<uint32_t>(2);                     // starting page offset
    [[maybe_unused]] uint32_t num_whole_packets = get_arg_val<uint32_t>(3);    // whole packets (fallback paths)
    [[maybe_unused]] uint32_t num_pages_remainder = get_arg_val<uint32_t>(4);  // remainder pages (fallback paths)

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_base_addr);

    if constexpr (enable_bank_packing) {
        // Interleaved bank-contiguous packing. Pages whose indices differ by num_banks live in the
        // same bank at consecutive slots, so they are contiguous in memory and can be gathered with a
        // single noc_async_read and forwarded in a single fabric packet covering
        // {p, p + num_banks, ..., p + (count - 1) * num_banks}.
        //
        // Each CB entry holds one super-block of (num_banks * num_pages_per_packet) pages. Within the
        // entry, bank b occupies a fixed region [b * bank_region_bytes]. We issue one read per bank
        // before a single barrier so the reads overlap across DRAM banks (preserving read
        // parallelism). The first num_banks consecutive pages each start a distinct bank, and
        // stepping by num_banks stays within that bank, so no per-iteration modulus is needed.
        constexpr uint32_t super_block_pages = num_banks * num_pages_per_packet;
        constexpr uint32_t bank_region_bytes = num_pages_per_packet * input_page_size;
        const uint32_t end_page = page_start_offset + num_pages;
        for (uint32_t sb_base = page_start_offset; sb_base < end_page; sb_base += super_block_pages) {
            cb_reserve_back(cb0_id, 1);
            const uint32_t l1_base = get_write_ptr(cb0_id);
            for (uint32_t b = 0; b < num_banks; ++b) {
                const uint32_t head = sb_base + b;
                if (head >= end_page) {
                    break;  // remaining banks in this super-block have no pages
                }
                uint32_t count = 0;
                for (uint32_t pp = head; count < num_pages_per_packet && pp < end_page; pp += num_banks) {
                    ++count;
                }
                noc_async_read<bank_region_bytes>(
                    input_addr_gen.get_noc_addr(head), l1_base + b * bank_region_bytes, count * input_page_size);
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
        }
    }
    // TODO #24995: Instead of page by page transfers, we can transfer bank by bank
    // Small pages. We pack multiple pages into a single packet.
    else if constexpr (num_pages_per_packet > 0) {
        uint32_t page_index = page_start_offset;
        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            cb_reserve_back(cb0_id, 1);
            auto l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t j = 0; j < num_pages_per_packet; ++j) {
                auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
                noc_async_read<input_page_size>(noc_read_addr, l1_write_addr, input_page_size);
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
        }

        if (num_pages_remainder > 0) {
            cb_reserve_back(cb0_id, 1);
            auto l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t j = 0; j < num_pages_remainder; ++j) {
                auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
                noc_async_read<input_page_size>(noc_read_addr, l1_write_addr, input_page_size);
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
        }

    }
    // Large pages. We pack page chunks into a single packet.
    else {
        // TODO #24995: Could read whole page into scratch, then copy locally
        uint32_t page_index = page_start_offset;
        for (uint32_t i = 0; i < num_pages; ++i) {
            auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
            for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
                cb_reserve_back(cb0_id, 1);
                auto l1_write_addr = get_write_ptr(cb0_id);
                noc_async_read<whole_packet_size>(noc_read_addr, l1_write_addr, whole_packet_size);
                noc_read_addr += whole_packet_size;
                noc_async_read_barrier();
                cb_push_back(cb0_id, 1);
            }
            if constexpr (partial_packet_size > 0) {
                cb_reserve_back(cb0_id, 1);
                auto l1_write_addr = get_write_ptr(cb0_id);
                noc_async_read<partial_packet_size>(noc_read_addr, l1_write_addr, partial_packet_size);
                noc_async_read_barrier();
                cb_push_back(cb0_id, 1);
            }
            page_index++;
        }
    }
}
