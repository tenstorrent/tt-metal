// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // total_size_bytes is derived from the tensor spec (aligned size per bank), NOT from any
    // buffer storage address, so it is safe to deliver via a named runtime arg: it stays valid
    // across program-cache hits with different storage of the same spec.
    uint32_t total_size_bytes = get_arg(args::total_size_bytes);

    Noc noc;

    // The resident input/output shards are reached by L1 base address from local TensorAccessors
    // (no borrowed self-loop DFBs, which Metal 2.0 forbids on DM kernels). These base addresses are
    // the real L1 addresses of those buffers, refreshed by the framework on every program-cache hit.
    // We deliberately recompute the move-chunk size from these base pointers in-kernel rather than
    // receiving the host-computed (output_addr - input_addr) delta via a runtime arg, which would go
    // stale on a cache hit with different storage and silently read/write the wrong addresses.
    uint32_t src_cb_base_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::input).get_noc_addr(0));
    uint32_t dst_cb_base_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::output).get_noc_addr(0));

    // The op only takes this (backwards, intra-L1, overlapping) path when the output buffer is at a
    // higher address than the input buffer, so the delta is strictly positive.
    uint32_t chunk_size_bytes = dst_cb_base_addr - src_cb_base_addr;

    // Copy from top of src cb to top of dst cb (backwards) in chunks of chunk_size_bytes, with a
    // final smaller chunk for any remainder. Computed with a subtraction loop instead of `/` and `%`
    // so this kernel does NOT pull in the libgcc software-divide routines (__udivsi3/__umodsi3).
    // On the Quasar XIP build those extra library routines perturb the crt0 layout and trip XIPify's
    // HI20/LO12 pairing (orphaned R_RISCV_HI20 for __tdata_lma -> tt_elffile.cpp TT_THROW). Behaviour
    // is identical to the prior num_chunks/remainder form for the valid (dst>src, chunk>0) inputs
    // this backwards-overlapping path is only ever taken with.
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    uint32_t bytes_remaining = total_size_bytes;
    while (bytes_remaining > 0) {
        uint32_t this_chunk_bytes = (bytes_remaining >= chunk_size_bytes) ? chunk_size_bytes : bytes_remaining;
        src_cb_addr -= this_chunk_bytes;
        dst_cb_addr -= this_chunk_bytes;
        CoreLocalMem<uint32_t> dst(dst_cb_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            this_chunk_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_cb_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        bytes_remaining -= this_chunk_bytes;
    }
}
