// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

// Writes this core's chunk outputs and its tile rows of the final state.
template <uint32_t num_chunks, uint32_t kt>
TT_KERNEL void writer(uint32_t head, uint32_t first_chunk) {
    DataflowBuffer out(dfb::out);
    DataflowBuffer final_rows(dfb::final_rows);
    Noc noc;

    // The reader's control words: this core's chunk count and whether it forms the final state.
    uint32_t chunks;
    uint32_t final;
    uint32_t first_row;
    uint32_t row_stride;
    {
        DataflowBuffer ctl(dfb::ctl);
        ctl.wait_front(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl.get_read_ptr());
        chunks = words[0];
        final = words[1];
        first_row = words[2];
        row_stride = words[3];
        ctl.pop_front(1);
    }

    const auto output = TensorAccessor(tensor::output);
    const uint32_t out_tile = out.get_entry_size();
    for (uint32_t c = first_chunk; c < first_chunk + chunks; ++c) {
        out.wait_front(kt);
        const uint32_t row = (head * num_chunks + c) * kt;
        for (uint32_t j = 0; j < kt; ++j) {
            noc.async_write(out, output, out_tile, {.offset_bytes = j * out_tile}, {.page_id = row + j});
        }
        noc.async_writes_flushed();
        out.pop_front(kt);
    }
    if (final) {
        const auto states = TensorAccessor(tensor::final_state);
        const uint32_t tile = final_rows.get_entry_size();
        // Row j of the state: tiles (j, c).
        for (uint32_t j = first_row; j < kt; j += row_stride) {
            final_rows.wait_front(kt);
            for (uint32_t c = 0; c < kt; ++c) {
                noc.async_write(
                    final_rows, states, tile, {.offset_bytes = c * tile}, {.page_id = head * kt * kt + j * kt + c});
            }
            noc.async_writes_flushed();
            final_rows.pop_front(kt);
        }
    }
    noc.async_write_barrier();
}
