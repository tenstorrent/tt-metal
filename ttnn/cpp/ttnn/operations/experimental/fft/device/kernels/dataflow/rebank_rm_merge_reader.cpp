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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_unit = get_arg(args::base_unit);
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t CHUNK = get_arg(args::chunk);
    constexpr uint32_t IS_BF16 = get_arg(args::is_bf16);

    constexpr uint32_t elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr uint32_t chunk_bytes = CHUNK * elem_bytes;

    const auto src_gen = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer block(dfb::block);

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t src_page = base_unit + u;  // sequential full-page reads

        block.reserve_back(1u);

        // Read the entire source page (CHUNK elements) at offset 0.
        noc.async_read(src_gen, block, chunk_bytes, {.page_id = src_page, .offset_bytes = 0u}, {});
        noc.async_read_barrier();

        block.push_back(1u);
    }
}
