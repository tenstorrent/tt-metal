// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Interleaved DRAM writer. Issues a whole block's tile-writes back-to-back and barriers once, instead of the
// stock tilize writer's per-tile flush (noc_async_writes_flushed). On the bf16 / DRAM-read-bound path this is
// bandwidth-neutral — the op's win is the padding skip, not the writer — but batching avoids the per-tile flush
// overhead and is harmless. The output CB holds exactly one block, so wait_front(tiles_per_block) always leaves
// the read pointer at the CB base (contiguous, valid src offset). Output is always interleaved DRAM
// (enforced in validate); there is no sharded path.
//
// On the skip path the compute produces only this_core_blocks blocks (the filled prefix), so the writer reads
// that count from the control CB (c_1) and writes exactly that many blocks — matching the compute.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_arg = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr bool skip_padding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t cb_ctl_id = tt::CBIndex::c_1;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    const uint32_t tiles_per_block = get_local_cb_interface(cb_id_out).fifo_num_pages;

    Noc noc;
    DataflowBuffer dfb(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr);

    uint32_t num_pages = num_pages_arg;
    if constexpr (skip_padding) {
        DataflowBuffer dfb_ctl(cb_ctl_id);
        dfb_ctl.wait_front(1);
        volatile tt_l1_ptr uint32_t* ctl = (volatile tt_l1_ptr uint32_t*)dfb_ctl.get_read_ptr();
        num_pages = ctl[0] * tiles_per_block;
    }

    uint32_t page = start_id;
    uint32_t remaining = num_pages;
    while (remaining > 0) {
        const uint32_t batch = (remaining < tiles_per_block) ? remaining : tiles_per_block;
        dfb.wait_front(batch);
        for (uint32_t k = 0; k < batch; ++k) {
            noc.async_write(dfb, s, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = page + k});
        }
        noc.async_write_barrier();
        dfb.pop_front(batch);
        page += batch;
        remaining -= batch;
    }
}
