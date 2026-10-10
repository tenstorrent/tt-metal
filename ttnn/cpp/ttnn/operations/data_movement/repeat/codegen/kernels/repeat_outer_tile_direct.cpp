// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Outer-axis TILE repeat: each source tile is read once and written to every copy.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);
    const uint32_t num_pages = get_arg_val<uint32_t>(3);
    const uint32_t repetitions = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t lower_pages = get_compile_time_arg_val(2);
    constexpr uint32_t rep_dim_pages = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t repeated_block = lower_pages * rep_dim_pages;

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);

    Noc noc;
    // The CB is this kernel's private staging slot, never handed to another RISC.
    CircularBuffer cb(cb_id);
    const CoreLocalMem<uint32_t> local(cb.get_write_ptr());

    const uint32_t end_page = start_page + num_pages;
    for (uint32_t page = start_page; page < end_page; ++page) {
        noc.async_read(src, local, page_bytes, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();

        const uint32_t higher = page / repeated_block;
        const uint32_t within = page - higher * repeated_block;
        const uint32_t output_base = higher * repeated_block * repetitions + within;
        for (uint32_t copy = 0; copy < repetitions; ++copy) {
            noc.async_write(
                local,
                dst,
                page_bytes,
                {.offset_bytes = 0},
                {.page_id = output_base + copy * repeated_block, .offset_bytes = 0});
        }
        // The next read overwrites the slot, so the writes must have left it, but need not have landed.
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}
