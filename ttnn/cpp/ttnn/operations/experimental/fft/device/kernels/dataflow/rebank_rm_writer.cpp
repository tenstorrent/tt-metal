// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// rebank_rm_writer.cpp — BRISC1 / writer for rebank_rm.
//
// Consumes CHUNK-element blocks from CB_REBANK (filled by the reader)
// and writes them to consecutive output rows of the destination tensor
// (B_total * N/CHUNK, CHUNK) with page_size = CHUNK * elem_bytes.
//
// Runtime args:
//   0: dst_addr              (DRAM buffer base address)
//   1: base_unit             (first work unit for this core)
//   2: num_units             (work units this core handles)
//   3: dst_page_size_bytes   (= CHUNK * elem_bytes)
//
// Compile-time args:
//   0: CHUNK   (elements per output row)
//   1: IS_BF16 (0 = fp32, 1 = bf16)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_REBANK = 0u;

void kernel_main() {
    const uint32_t dst_addr            = get_arg_val<uint32_t>(0);
    const uint32_t base_unit           = get_arg_val<uint32_t>(1);
    const uint32_t num_units           = get_arg_val<uint32_t>(2);
    const uint32_t dst_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t CHUNK  = get_compile_time_arg_val(0);
    constexpr uint32_t IS_BF16 = get_compile_time_arg_val(1);

    constexpr uint32_t elem_bytes  = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr, .page_size = dst_page_size_bytes};

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t dst_row = base_unit + u;

        cb_wait_front(CB_REBANK, 1u);
        const uint32_t l1_ptr = get_read_ptr(CB_REBANK);

        const uint64_t noc_addr = dst_gen.get_noc_addr(dst_row, 0u);
        noc_async_write(l1_ptr, noc_addr, chunk_bytes);
        noc_async_write_barrier();

        cb_pop_front(CB_REBANK, 1u);
    }
}
