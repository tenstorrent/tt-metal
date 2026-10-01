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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_unit = get_arg(args::base_unit);
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t CHUNK = get_arg(args::chunk);
    constexpr uint32_t CHUNKS_PER_ROW = get_arg(args::chunks_per_row);
    constexpr uint32_t IS_BF16 = get_arg(args::is_bf16);

    constexpr uint32_t elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    // CHUNKS_PER_ROW is a compile-time constant; the compiler can optimise
    // division and modulo (e.g. via reciprocal multiplication) whether or not
    // CHUNKS_PER_ROW is a power of 2.
    const auto src_gen = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer block(dfb::block);

    // Starting page and chunk-within-page for base_unit.
    uint32_t src_page = base_unit / CHUNKS_PER_ROW;
    uint32_t chunk_in_page = base_unit % CHUNKS_PER_ROW;

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t col_offset = chunk_in_page * chunk_bytes;

        block.reserve_back(1u);

        // Single contiguous read of CHUNK elements — never crosses a page.
        noc.async_read(
            src_gen,
            block,
            chunk_bytes,
            {.page_id = src_page, .offset_bytes = col_offset},
            {});
        noc.async_read_barrier();

        block.push_back(1u);

        // Advance position within the source row.
        if (++chunk_in_page == CHUNKS_PER_ROW) {
            chunk_in_page = 0u;
            ++src_page;
        }
    }
}
