// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Generic writer for the fused-QKV paths: the compute result (Nt tiles) lands in out_cb; write it
// to a DRAM-interleaved output starting at page `start_page` (the flat output is a single tile-row,
// so page == tile column index). KV uses start_page=0; each Q core uses its head-slice offset.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    Noc noc;
    CircularBuffer out_cb_obj(out_cb);
    const uint32_t out_tile_bytes = get_tile_size(out_cb);
    const auto s_out = TensorAccessor(out_args, out_addr);

    out_cb_obj.wait_front(Nt);
    for (uint32_t nt = 0; nt < Nt; ++nt) {
        noc.async_write(
            out_cb_obj, s_out, out_tile_bytes, {.offset_bytes = nt * out_tile_bytes}, {.page_id = start_page + nt});
    }
    noc.async_write_barrier();
    out_cb_obj.pop_front(Nt);
}
