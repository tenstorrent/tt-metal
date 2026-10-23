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

    const auto dst_gen = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer block(dfb::block);

    for (uint32_t u = 0u; u < num_units; ++u) {
        const uint32_t dst_row = base_unit + u;

        block.wait_front(1u);

        noc.async_write(block, dst_gen, chunk_bytes, {}, {.page_id = dst_row, .offset_bytes = 0u});
        noc.async_write_barrier();

        block.pop_front(1u);
    }
}
