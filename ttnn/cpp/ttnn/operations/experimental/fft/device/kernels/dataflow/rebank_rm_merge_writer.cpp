// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// rebank_rm_merge_writer.cpp — BRISC1 / writer for rebank_rm_merge.
//
// Consumes CHUNK-element blocks from CB_MERGE (filled by the reader) and
// writes them at the correct byte offset within the large destination page:
//
//   dst_page   = (base_unit + u) / CHUNKS_PER_MERGE
//   col_offset = ((base_unit + u) % CHUNKS_PER_MERGE) * CHUNK * elem_bytes
//
// CHUNKS_PER_MERGE is a compile-time constant; the compiler optimises division
// and modulo (e.g. via reciprocal multiplication) for any integer value.
//
// Runtime args:
//   0: dst_addr              (DRAM buffer base address)
//   1: base_unit             (first work unit for this core)
//   2: num_units             (work units this core handles)
//   3: dst_page_size_bytes   (= CHUNK * CHUNKS_PER_MERGE * elem_bytes, large)
//
// Compile-time args:
//   0: CHUNK             (elements per source row = write granularity)
//   1: CHUNKS_PER_MERGE  (source rows merged into one output row; any integer ≥ 1)
//   2: IS_BF16           (0 = fp32, 1 = bf16)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_MERGE = 0u;

void kernel_main() {
    const uint32_t dst_addr            = get_arg_val<uint32_t>(0);
    const uint32_t base_unit           = get_arg_val<uint32_t>(1);
    const uint32_t num_units           = get_arg_val<uint32_t>(2);
    const uint32_t dst_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t CHUNK            = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNKS_PER_MERGE = get_compile_time_arg_val(1);
    constexpr uint32_t IS_BF16          = get_compile_time_arg_val(2);

    constexpr uint32_t elem_bytes  = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr, .page_size = dst_page_size_bytes};

    // Starting page and chunk-within-page for base_unit.
    uint32_t dst_page      = base_unit / CHUNKS_PER_MERGE;
    uint32_t chunk_in_page = base_unit % CHUNKS_PER_MERGE;

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t col_offset = chunk_in_page * chunk_bytes;

        cb_wait_front(CB_MERGE, 1u);
        const uint32_t l1_ptr = get_read_ptr(CB_MERGE);

        // Write CHUNK elements at the correct byte offset within the large page.
        const uint64_t noc_addr = dst_gen.get_noc_addr(dst_page, col_offset);
        noc_async_write(l1_ptr, noc_addr, chunk_bytes);
        noc_async_write_barrier();

        cb_pop_front(CB_MERGE, 1u);

        // Advance position within the destination row.
        if (++chunk_in_page == CHUNKS_PER_MERGE) {
            chunk_in_page = 0u;
            ++dst_page;
        }
    }
}
