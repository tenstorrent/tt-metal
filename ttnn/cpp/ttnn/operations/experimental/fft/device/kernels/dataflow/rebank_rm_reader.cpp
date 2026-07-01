// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// rebank_rm_reader.cpp — BRISC0 / reader for rebank_rm.
//
// rebank_rm converts an (B_total, N) ROW_MAJOR tensor whose page_size is
// N*elem_bytes (one large page per batch row) into an
// (B_total * N/CHUNK, CHUNK) tensor with page_size = CHUNK*elem_bytes.
// The operation is a pure page-boundary-aware copy: no transposition,
// no arithmetic.
//
// Each work unit u ∈ [base_unit, base_unit + num_units) corresponds to
// one output row of CHUNK elements:
//   src_page   = u / CHUNKS_PER_ROW    (which source batch row)
//   col_offset = (u % CHUNKS_PER_ROW) * CHUNK * elem_bytes
//
// Because CHUNK divides N exactly (CHUNK ≤ N and N % CHUNK == 0),
// the read never crosses a source page boundary.  One NoC read per unit.
// N need not be a power of 2; CHUNKS_PER_ROW (= N/CHUNK) may be any integer.
//
// Runtime args:
//   0: src_addr              (DRAM buffer base address)
//   1: base_unit             (first work unit for this core)
//   2: num_units             (work units this core handles)
//   3: src_page_size_bytes   (= N * elem_bytes)
//
// Compile-time args:
//   0: CHUNK           (target last-dim size in elements; must be pow-2)
//   1: CHUNKS_PER_ROW  (= N / CHUNK; must be pow-2)
//   2: IS_BF16         (0 = fp32, 1 = bf16)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_REBANK = 0u;

void kernel_main() {
    const uint32_t src_addr            = get_arg_val<uint32_t>(0);
    const uint32_t base_unit           = get_arg_val<uint32_t>(1);
    const uint32_t num_units           = get_arg_val<uint32_t>(2);
    const uint32_t src_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t CHUNK          = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNKS_PER_ROW = get_compile_time_arg_val(1);
    constexpr uint32_t IS_BF16        = get_compile_time_arg_val(2);

    constexpr uint32_t elem_bytes  = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    // CHUNKS_PER_ROW is a compile-time constant; the compiler can optimise
    // division and modulo (e.g. via reciprocal multiplication) whether or not
    // CHUNKS_PER_ROW is a power of 2.
    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr, .page_size = src_page_size_bytes};

    // Starting page and chunk-within-page for base_unit.
    uint32_t src_page      = base_unit / CHUNKS_PER_ROW;
    uint32_t chunk_in_page = base_unit % CHUNKS_PER_ROW;

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t col_offset = chunk_in_page * chunk_bytes;

        cb_reserve_back(CB_REBANK, 1u);
        const uint32_t l1_ptr = get_write_ptr(CB_REBANK);

        // Single contiguous read of CHUNK elements — never crosses a page.
        const uint64_t noc_addr = src_gen.get_noc_addr(src_page, col_offset);
        noc_async_read(noc_addr, l1_ptr, chunk_bytes);
        noc_async_read_barrier();

        cb_push_back(CB_REBANK, 1u);

        // Advance position within the source row.
        if (++chunk_in_page == CHUNKS_PER_ROW) {
            chunk_in_page = 0u;
            ++src_page;
        }
    }
}
