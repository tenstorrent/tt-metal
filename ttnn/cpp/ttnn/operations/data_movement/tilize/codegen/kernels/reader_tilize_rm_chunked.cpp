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

void kernel_main() {
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t H_per_tile = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t chunk_Wt = get_named_compile_time_arg_val("chunk_wt");
    constexpr uint32_t num_col_chunks = get_named_compile_time_arg_val("num_col_chunks");
    constexpr uint32_t elem_w_bytes = get_named_compile_time_arg_val("fragment_bytes");  // tile width * element size
    constexpr uint32_t aligned_page_size = get_named_compile_time_arg_val("aligned_page_size");
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t chunk_read_bytes = chunk_Wt * elem_w_bytes;

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t start_stick = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(src_args, src_addr, aligned_page_size);
    uint32_t i_stick = start_stick;

    Noc noc;
    CircularBuffer cb(cb_id);

    if constexpr (num_col_chunks == 1) {
        for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
            cb.reserve_back(chunk_Wt);
            uint32_t l1_offset = 0;
            for (uint32_t h = 0; h < H_per_tile; ++h) {
                noc.async_read(
                    s, cb, chunk_read_bytes, {.page_id = i_stick, .offset_bytes = 0}, {.offset_bytes = l1_offset});
                l1_offset += chunk_read_bytes;
                i_stick++;
            }
            noc.async_read_barrier();
            cb.push_back(chunk_Wt);
        }
    } else {
        // Resolve each stick's NoC address once per tile row, not per chunk.
        const uint8_t noc_id = noc.get_noc_id();
        for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
            uint64_t base_noc[H_per_tile];
            for (uint32_t h = 0; h < H_per_tile; ++h) {
                base_noc[h] = s.get_noc_addr(i_stick + h, 0, noc_id);
            }
            uint32_t chunk_off = 0;
            for (uint32_t c = 0; c < num_col_chunks; ++c) {
                cb.reserve_back(chunk_Wt);
                const CoreLocalMem<uint8_t> dst(cb.get_write_ptr());
                uint32_t l1_offset = 0;
                for (uint32_t h = 0; h < H_per_tile; ++h) {
                    noc.async_read(
                        tensor_accessor::Page(base_noc[h], i_stick + h),
                        dst,
                        chunk_read_bytes,
                        {.offset_bytes = chunk_off},
                        {.offset_bytes = l1_offset});
                    l1_offset += chunk_read_bytes;
                }
                noc.async_read_barrier();
                cb.push_back(chunk_Wt);
                chunk_off += chunk_read_bytes;
            }
            i_stick += H_per_tile;
        }
    }
}
