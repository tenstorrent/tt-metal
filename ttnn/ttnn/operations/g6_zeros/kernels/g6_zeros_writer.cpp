// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_zeros writer — the op's only kernel (no reader, no compute: the op has no input and no math).
//
// Regime R1 (bank-affine coalesced runs). This core owns `run_pages` contiguous bank-rows of ONE
// DRAM bank, starting at page `first_page`. Interleaved bank-rows of one bank are contiguous bytes
// `aps` apart, so the whole run is one contiguous DRAM byte range, written in
// ceil(run_pages / burst_pages) NoC writes of <= burst_pages * aps bytes from one L1 zero source.
//
// Block schedule (block = this core's whole run):
//   fill_zero_source  — once: loopback zero-fill of cb_zero_source (Noc::async_write_zeros, overload 1)
//   store_zero_run    — raw noc_async_write per burst, no barrier inside the run
//   drain_writes      — once: noc_async_write_barrier
//
// Raw-API note for store_zero_run: Noc::async_write_zeros overload 2 (TensorAccessor) is limited to a
// single page per call (noc_zero_dram.inl ASSERT offset+size <= aligned page size). The planned block
// is a multi-page bank-contiguous run, so the burst write is a raw noc_async_write whose byte count is
// bounded by the zero source size (burst_pages * aps) by construction.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t num_banks = get_compile_time_arg_val(0);          // B (host allocator view)
    constexpr uint32_t burst_pages = get_compile_time_arg_val(1);        // pages per NoC write
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);  // aps
    constexpr uint32_t cb_zero_source_id = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    constexpr uint32_t burst_bytes = burst_pages * aligned_page_size;  // == zero_source_bytes

    // ---- runtime args ----
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t run_pages = get_arg_val<uint32_t>(1);

    // Ragged tail of the core grid: a core with no bank-rows does nothing.
    const uint32_t num_blocks_this_core = run_pages > 0 ? 1 : 0;
    if (num_blocks_this_core == 0) {
        return;
    }

    const auto output = TensorAccessor(output_args, output_addr, aligned_page_size);
    ASSERT(output.contiguous_page_stride() == num_banks);

    Noc noc;
    CircularBuffer cb_zero_source(cb_zero_source_id);

    {
        DeviceZoneScopedN("fill_zero_source");
        // Fresh CB: read ptr == write ptr; never pushed/popped — a raw same-thread scratch.
        noc.async_write_zeros(cb_zero_source, burst_bytes);
        noc.write_zeros_l1_barrier();
    }
    const uint32_t zero_src_l1 = cb_zero_source.get_write_ptr();

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        DeviceZoneScopedN("store_zero_run");
        ASSERT(output.num_contiguous_pages(first_page, first_page + (run_pages - 1) * num_banks + 1) == run_pages);

        uint64_t dst_noc = output.get_noc_addr(first_page);
        uint32_t remaining = run_pages;
        while (remaining >= burst_pages) {
            noc_async_write(zero_src_l1, dst_noc, burst_bytes);
            dst_noc += burst_bytes;
            remaining -= burst_pages;
        }
        if (remaining > 0) {
            noc_async_write(zero_src_l1, dst_noc, remaining * aligned_page_size);
        }
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
