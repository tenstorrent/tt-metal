// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros writer (BRISC, NoC1). The op's only kernel: there is no input to read and no arithmetic.
//
// Raw dataflow API, no kernel_lib helper: zero_tile()/prepare_zero_tile() zero exactly one tile per
// call with a barrier each, and Noc::async_write_zeros overload 2 stays inside one page
// (noc_zero_dram.inl). A bank run spans many pages of one bank, so it is written here in
// zero_source_bytes chunks from one core-local zero source.
//
// Interleaved DRAM: page p sits in bank p % NUM_DRAM_BANKS at slot p / NUM_DRAM_BANKS, and the
// slots of one bank are contiguous. A block is one bank's run of consecutive slots: one contiguous
// DRAM region, written back to back with no barrier between writes.

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_zero_source = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t zero_source_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(3);
    constexpr uint32_t banks_per_core = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    // The host groups pages by its own bank count; the accessor strides by NUM_DRAM_BANKS.
    static_assert(num_dram_banks == NUM_DRAM_BANKS, "host and device disagree on the DRAM bank count");
    static_assert(zero_source_bytes % 64 == 0, "zero source must keep every DRAM chunk 64 B aligned");

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t bank_set_base = get_arg_val<uint32_t>(0);
    const uint32_t first_bank_offset = get_arg_val<uint32_t>(1);
    const uint32_t slot_begin = get_arg_val<uint32_t>(2);
    const uint32_t core_slots = get_arg_val<uint32_t>(3);
    const uint32_t block_slots = get_arg_val<uint32_t>(4);
    const uint32_t slot_end = slot_begin + core_slots;

    // page_bytes is the aligned page size, so num_contiguous_pages reports whole bank runs.
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);

    Noc noc;
    CircularBuffer zero_source(cb_zero_source);
    // A never-reserved CB has read_ptr == write_ptr: the fill target is also the write source.
    const uint32_t zero_source_addr = zero_source.get_write_ptr();

    {
        DeviceZoneScopedN("ZEROS-FILL");
        noc.async_write_zeros(zero_source, zero_source_bytes);
        noc.write_zeros_l1_barrier();
    }

    {
        DeviceZoneScopedN("ZEROS-WRITE");
        for (uint32_t bank_k = 0; bank_k < banks_per_core; ++bank_k) {
            const uint32_t bank = bank_set_base + (first_bank_offset + bank_k) % banks_per_core;
            for (uint32_t slot = slot_begin; slot < slot_end; slot += block_slots) {
                const uint32_t first_page = slot * num_dram_banks + bank;
                if (first_page >= tensor_pages) {
                    break;
                }
                const uint32_t block_pages = std::min(block_slots, slot_end - slot);
                const uint32_t run_pages = std::min(block_pages, output.num_contiguous_pages(first_page, tensor_pages));
                const uint32_t run_bytes = run_pages * page_bytes;
                const uint64_t run_noc_addr = output.get_noc_addr(first_page);
                for (uint32_t written = 0; written < run_bytes; written += zero_source_bytes) {
                    const uint32_t chunk = std::min(zero_source_bytes, run_bytes - written);
                    noc_async_write(zero_source_addr, run_noc_addr + written, chunk);
                }
            }
        }
    }

    {
        DeviceZoneScopedN("ZEROS-DRAIN");
        noc_async_write_barrier();
    }
}
