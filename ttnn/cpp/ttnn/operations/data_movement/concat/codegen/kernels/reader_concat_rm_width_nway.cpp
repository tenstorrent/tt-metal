// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// N-way ROW_MAJOR width-concat reader.
//
// Each output stick is assembled directly from all original input sticks.  No
// pairwise intermediate is materialized, which avoids both extra dispatches and
// the shape-dependent corruption seen when an unaligned intermediate width is
// fed into another two-input concat on Blackhole.
//
// All inputs share one TensorAccessor ABI (same interleaved memory placement),
// but carry independent logical and aligned page sizes.
//
// CT args:
//   cb_out, cb_scratch, N_INPUTS, OUT_PAGE_SIZE, BATCH,
//   TensorAccessorArgs(shared input placement)
// RT args:
//   num_pages, start_page,
//   bases[N_INPUTS], stick_sizes[N_INPUTS], page_sizes[N_INPUTS]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    uint32_t src_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t N_INPUTS = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t BATCH = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t base_rt = 2;
    constexpr uint32_t stick_rt = base_rt + N_INPUTS;
    constexpr uint32_t page_rt = stick_rt + N_INPUTS;

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer scratch_cb(cb_scratch);

    scratch_cb.reserve_back(1);
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();

    while (num_pages > 0) {
        const uint32_t batch = num_pages < BATCH ? num_pages : BATCH;
        out_cb.reserve_back(batch);
        const uint32_t out_base = out_cb.get_write_ptr();

        for (uint32_t page = 0; page < batch; ++page) {
            uint32_t dst = out_base + page * OUT_PAGE_SIZE;
            uint32_t dst_offset = 0;

            for (uint32_t input = 0; input < N_INPUTS; ++input) {
                const uint32_t base = get_arg_val<uint32_t>(base_rt + input);
                const uint32_t stick_size = get_arg_val<uint32_t>(stick_rt + input);
                const uint32_t page_size = get_arg_val<uint32_t>(page_rt + input);
                const auto accessor = TensorAccessor(src_args, base, page_size);

                // Physical pages are always NOC-aligned.  Read one complete
                // source page, then copy only its logical bytes into the packed
                // output stick.  A byte loop handles bf16 and any future RM dtype
                // without imposing alignment on mixed-width segment boundaries.
                noc.async_read(accessor, scratch_cb, page_size, {.page_id = src_page + page}, {.offset_bytes = 0});
                noc.async_read_barrier();
                CoreLocalMem<volatile uint8_t> src(scratch_addr);
                CoreLocalMem<volatile uint8_t> target(dst + dst_offset);
                for (uint32_t byte = 0; byte < stick_size; ++byte) {
                    target[byte] = src[byte];
                }
                dst_offset += stick_size;
            }
        }

        out_cb.push_back(batch);
        src_page += batch;
        num_pages -= batch;
    }
}
