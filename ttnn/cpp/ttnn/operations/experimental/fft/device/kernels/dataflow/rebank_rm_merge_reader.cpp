// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// rebank_rm_merge_reader.cpp — BRISC0 / reader for rebank_rm_merge.
//
// rebank_rm_merge is the inverse of rebank_rm: it converts a
// (B_total * CHUNKS_PER_MERGE, CHUNK) ROW_MAJOR tensor with
// page_size = CHUNK*elem_bytes into a (B_total, CHUNK*CHUNKS_PER_MERGE)
// tensor with page_size = CHUNK*CHUNKS_PER_MERGE*elem_bytes.
//
// Unlike rebank_rm, the source pages are SMALL (CHUNK elements each).
// Each work unit u corresponds to reading one full source page:
//   src_page = base_unit + u      (sequential source row)
//   col_offset = 0                (always reads the full source page)
//
// The writer (rebank_rm_merge_writer.cpp) handles placing the data
// at the correct byte offset within the large destination page.
//
// Runtime args:
//   0: src_addr              (DRAM buffer base address)
//   1: base_unit             (first work unit for this core)
//   2: num_units             (work units this core handles)
//   3: src_page_size_bytes   (= CHUNK * elem_bytes, small)
//
// Compile-time args:
//   0: CHUNK           (source last-dim = number of elements per source row)
//   1: IS_BF16         (0 = fp32, 1 = bf16)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_MERGE = 0u;

void kernel_main() {
    const uint32_t src_addr            = get_arg_val<uint32_t>(0);
    const uint32_t base_unit           = get_arg_val<uint32_t>(1);
    const uint32_t num_units           = get_arg_val<uint32_t>(2);
    const uint32_t src_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t CHUNK  = get_compile_time_arg_val(0);
    constexpr uint32_t IS_BF16 = get_compile_time_arg_val(1);

    constexpr uint32_t elem_bytes  = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr, .page_size = src_page_size_bytes};

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t src_page = base_unit + u;  // sequential full-page reads

        cb_reserve_back(CB_MERGE, 1u);
        const uint32_t l1_ptr = get_write_ptr(CB_MERGE);

        // Read the entire source page (CHUNK elements) at offset 0.
        const uint64_t noc_addr = src_gen.get_noc_addr(src_page, 0u);
        noc_async_read(noc_addr, l1_ptr, chunk_bytes);
        noc_async_read_barrier();

        cb_push_back(CB_MERGE, 1u);
    }
}
