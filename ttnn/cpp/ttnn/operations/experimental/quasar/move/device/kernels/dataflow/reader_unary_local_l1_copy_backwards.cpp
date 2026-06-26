// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // total_size_bytes is derived from the tensor spec (aligned size per bank), NOT from any
    // buffer storage address, so it is safe to deliver via a named runtime arg: it stays valid
    // across program-cache hits with different storage of the same spec.
    uint32_t total_size_bytes = get_arg(args::total_size_bytes);

    Noc noc;

    // Both DFBs are borrowed onto the actual tensor buffers (src_cb <- input, dst_cb <- output),
    // so their base pointers are the real L1 addresses of those buffers and are refreshed by the
    // framework on every program-cache hit.  We deliberately recompute the move-chunk size from
    // these refreshed base pointers in-kernel rather than receiving the host-computed
    // (output_addr - input_addr) delta via a runtime arg, which would go stale on a cache hit with
    // different storage and silently read/write the wrong addresses.
    DataflowBuffer src_cb(dfb::src_cb);
    DataflowBuffer dst_cb(dfb::dst_cb);

    uint32_t src_cb_base_addr = src_cb.get_read_ptr();
    uint32_t dst_cb_base_addr = dst_cb.get_write_ptr();

    // The op only takes this (backwards, intra-L1, overlapping) path when the output buffer is at a
    // higher address than the input buffer, so the delta is strictly positive.
    uint32_t chunk_size_bytes = dst_cb_base_addr - src_cb_base_addr;
    uint32_t num_chunks = total_size_bytes / chunk_size_bytes;
    uint32_t remainder_chunk_size_bytes = total_size_bytes % chunk_size_bytes;

    // Copy from top of src cb to top of dst cb (backwards)
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    for (uint32_t i = 0; i < num_chunks; i += 1) {
        src_cb_addr -= chunk_size_bytes;
        dst_cb_addr -= chunk_size_bytes;
        CoreLocalMem<uint32_t> dst(dst_cb_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            chunk_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_cb_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
    }
    if (remainder_chunk_size_bytes > 0) {
        src_cb_addr -= remainder_chunk_size_bytes;
        dst_cb_addr -= remainder_chunk_size_bytes;
        CoreLocalMem<uint32_t> dst(dst_cb_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            remainder_chunk_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_cb_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
    }
}
