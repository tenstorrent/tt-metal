// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"

struct Args {
    uint32_t src_addr;
    uint32_t num_tile_rows;
    uint32_t start_stick;
    uint32_t H_per_tile;      // tile height
    uint32_t chunk_Wt;        // tiles per column chunk
    uint32_t num_col_chunks;  // chunks per tile row
    uint32_t elem_w_bytes;    // tile width * element size
};

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const auto* a = reinterpret_cast<const Args*>(get_arg_addr(0));
    const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);
    const uint32_t chunk_read_bytes = a->chunk_Wt * a->elem_w_bytes;
    uint32_t i_stick = a->start_stick;

    Noc noc;
    CircularBuffer cb(cb_id);

    if (a->num_col_chunks == 1) {
        for (uint32_t tr = 0; tr < a->num_tile_rows; ++tr) {
            cb.reserve_back(a->chunk_Wt);
            uint32_t l1_offset = 0;
            for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                noc.async_read(
                    s, cb, chunk_read_bytes, {.page_id = i_stick, .offset_bytes = 0}, {.offset_bytes = l1_offset});
                l1_offset += chunk_read_bytes;
                i_stick++;
            }
            noc.async_read_barrier();
            cb.push_back(a->chunk_Wt);
        }
    } else {
        // Each source stick is read once per column chunk at the same page, so its NoC address is resolved once
        // per tile row instead of once per chunk; the page-to-bank division is the expensive part. This is exact:
        // TensorAccessor adds the byte offset as the last term, so get_noc_addr(p, k) == get_noc_addr(p, 0) + k.
        // base_noc holds H_per_tile entries, which the factory always sets to the 32-row tile height.
        const uint8_t noc_id = noc.get_noc_id();
        for (uint32_t tr = 0; tr < a->num_tile_rows; ++tr) {
            uint64_t base_noc[32];
            for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                base_noc[h] = s.get_noc_addr(i_stick + h, 0, noc_id);
            }
            uint32_t chunk_off = 0;
            for (uint32_t c = 0; c < a->num_col_chunks; ++c) {
                cb.reserve_back(a->chunk_Wt);
                // The write pointer is fixed between reserve_back and push_back, so read it once per chunk rather
                // than on every transaction.
                const CoreLocalMem<uint8_t> dst(cb.get_write_ptr());
                uint32_t l1_offset = 0;
                for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                    noc.async_read(
                        tensor_accessor::Page(base_noc[h], i_stick + h),
                        dst,
                        chunk_read_bytes,
                        {.offset_bytes = chunk_off},
                        {.offset_bytes = l1_offset});
                    l1_offset += chunk_read_bytes;
                }
                noc.async_read_barrier();
                cb.push_back(a->chunk_Wt);
                chunk_off += chunk_read_bytes;
            }
            i_stick += a->H_per_tile;
        }
    }
}
