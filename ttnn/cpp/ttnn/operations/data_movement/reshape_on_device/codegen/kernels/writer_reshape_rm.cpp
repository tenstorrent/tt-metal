// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for RM reshape: sequential stick writes with new page size.
// Uses aligned_page_size for TensorAccessor addressing, new_stick_size for transfer.
// Matches tt-metal writer_unary_reshape_stick_layout_interleaved_multi_core.cpp
//
// CT args: cb_out, new_stick_size, aligned_page_size, TensorAccessorArgs(out_t),
//          PARTIAL (0/1), partial_bytes, noc_max_burst_bytes
// RT args: dst_addr, num_reads, num_sticks_per_read, num_sticks_per_cb_push, start_stick
//          [, col_off]   (col_off read only when PARTIAL)
//
// PARTIAL branch: the degenerate-gather parallelization. Multiple cores each
// assemble a `partial_bytes` slab of ONE output page and write it at an aligned
// column offset `col_off`. partial_bytes and col_off are DRAM-aligned by the
// spec.py gate, so disjoint cores hit disjoint byte ranges of the SAME page with
// no race and no cross-page spill. PARTIAL==0 is byte-identical to the original.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dst_addr              = get_arg_val<uint32_t>(0);
    uint32_t num_reads             = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_read   = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_stick           = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t new_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t PARTIAL = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr uint32_t partial_bytes = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t noc_max_burst_bytes = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 2);

    const auto s = TensorAccessor(dst_args, dst_addr, aligned_page_size);

    Noc noc;
    CircularBuffer out_buffer(cb_out0);

    if constexpr (PARTIAL == 0) {
        uint32_t i_stick = start_stick;
        for (uint32_t iter = 0; iter < num_reads; ++iter) {
            out_buffer.wait_front(num_sticks_per_cb_push);
            uint32_t l1_read_offset = 0;

            for (uint32_t i = 0; i < num_sticks_per_read; ++i) {
                // A single logical RM stick can exceed the architecture's NOC
                // transaction ceiling (WH nightly caught 36,952B vs 8,192B).
                // Chunk it without changing the page/CB layout.
                uint32_t remaining = new_stick_size;
                uint32_t offset = 0;
                while (remaining > 0) {
                    uint32_t burst = remaining < noc_max_burst_bytes
                        ? remaining : noc_max_burst_bytes;
                    noc.async_write(out_buffer, s, burst,
                                    {.offset_bytes = l1_read_offset + offset},
                                    {.page_id = i_stick, .offset_bytes = offset});
                    remaining -= burst;
                    offset += burst;
                }
                l1_read_offset += new_stick_size;
                i_stick += 1;
            }
            noc.async_write_barrier();
            out_buffer.pop_front(num_sticks_per_cb_push);
        }
    } else if constexpr (PARTIAL == 1) {
        // Degenerate-gather partial write: one CB page holds this core's
        // `partial_bytes` slab for output page `start_stick`; write it at the
        // aligned column offset `col_off`.
        uint32_t col_off = get_arg_val<uint32_t>(5);
        out_buffer.wait_front(1);
        uint32_t remaining = partial_bytes;
        uint32_t offset = 0;
        while (remaining > 0) {
            uint32_t burst = remaining < noc_max_burst_bytes
                ? remaining : noc_max_burst_bytes;
            noc.async_write(out_buffer, s, burst,
                            {.offset_bytes = offset},
                            {.page_id = start_stick, .offset_bytes = col_off + offset});
            remaining -= burst;
            offset += burst;
        }
        noc.async_write_barrier();
        out_buffer.pop_front(1);
    } else {
        // PARTIAL == 2 — degenerate-SCATTER slab write: ONE CB page holds this
        // core's num_sticks_per_read output-stick values, each at an L1_ALIGN-
        // aligned slot (stride = `partial_bytes` CT slot, = L1_ALIGN). The reader
        // (MODE_PARTIAL_READ) spread them there so each per-stick noc.async_write
        // meets the (src_L1 % L1_ALIGN) == (dst % L1_ALIGN) NOC rule — a packed
        // slab (stride new_stick_size < L1_ALIGN) floors the L1 source to the 16B
        // boundary and silently replicates. This is the symmetric partner of the
        // gather PARTIAL==1 path; the normal PARTIAL==0 writer can't be used
        // because its CB slots are new_aligned-padded per stick, not this stride.
        constexpr uint32_t slot_stride = partial_bytes;  // = L1_ALIGN (16)
        out_buffer.wait_front(1);
        uint32_t l1_read_offset = 0;
        uint32_t i_stick = start_stick;
        for (uint32_t i = 0; i < num_sticks_per_read; ++i) {
            noc.async_write(out_buffer, s, new_stick_size,
                            {.offset_bytes = l1_read_offset},
                            {.page_id = i_stick, .offset_bytes = 0});
            l1_read_offset += slot_stride;
            i_stick += 1;
        }
        noc.async_write_barrier();
        out_buffer.pop_front(1);
    }
}
