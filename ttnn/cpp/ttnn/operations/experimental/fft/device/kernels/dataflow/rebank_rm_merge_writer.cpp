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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_unit = get_arg(args::base_unit);
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t CHUNK = get_arg(args::chunk);
    constexpr uint32_t CHUNKS_PER_MERGE = get_arg(args::chunks_per_merge);
    constexpr uint32_t IS_BF16 = get_arg(args::is_bf16);

    constexpr uint32_t elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    const auto dst_gen = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer block(dfb::block);

    // Starting page and chunk-within-page for base_unit.
    uint32_t dst_page = base_unit / CHUNKS_PER_MERGE;
    uint32_t chunk_in_page = base_unit % CHUNKS_PER_MERGE;

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t col_offset = chunk_in_page * chunk_bytes;

        block.wait_front(1u);

        // Write CHUNK elements at the correct byte offset within the large page.
        noc.async_write(
            block,
            dst_gen,
            chunk_bytes,
            {},
            {.page_id = dst_page, .offset_bytes = col_offset});
        noc.async_write_barrier();

        block.pop_front(1u);

        // Advance position within the destination row.
        if (++chunk_in_page == CHUNKS_PER_MERGE) {
            chunk_in_page = 0u;
            ++dst_page;
        }
    }
}
