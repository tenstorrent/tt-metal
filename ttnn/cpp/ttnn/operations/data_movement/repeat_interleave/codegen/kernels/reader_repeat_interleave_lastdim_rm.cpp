// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for repeat_interleave on the LAST (within-stick, W) dim, RM interleaved.
//
// torch.repeat_interleave(x, R, dim=-1): each element along W is replicated R
// times consecutively (AABB), so out[j] = in[j / R] within every stick. Output
// sticks are 1:1 with input sticks (only W grows: W_out = W_in * R), so the page
// map is the identity (src_page == out_page); all the work is the intra-stick
// element expansion.
//
// Single-CB, no scratch: read the input stick into the TAIL of the (larger)
// output CB slot, then expand front-to-back IN PLACE. This never clobbers an
// unread source element -- for element i the write region ends at (i+1)*R and
// the source of any later element j>i sits at W_in*(R-1)+j >= (i+1)*R, with
// equality only at the last element (already read before it is overwritten).
//
// Block-float (bf8_b/bf4_b) is gated out by the Python builder: per-element copy
// would split the shared-exponent tile header (silent-wrong).
//
// CT: stick_out_bytes, stick_in_bytes, W_in, REPEATS, ELEM_SIZE,
//     in_read_size, input_page_size, input_stage_min_offset, input_alignment,
//     l1_slot_stride,
//     TensorAccessorArgs(in_t), cb_id, BATCH
// RT: src_addr, num_out_pages, out_start_page   (src_page == out_page)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_out_pages = get_arg_val<uint32_t>(1);
    uint32_t out_start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t stick_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_in = get_compile_time_arg_val(1);
    constexpr uint32_t W_in = get_compile_time_arg_val(2);
    constexpr uint32_t REPEATS = get_compile_time_arg_val(3);
    constexpr uint32_t ELEM_SIZE = get_compile_time_arg_val(4);
    constexpr uint32_t in_read_size = get_compile_time_arg_val(5);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t src_min_off = get_compile_time_arg_val(7);
    constexpr uint32_t input_alignment = get_compile_time_arg_val(8);
    constexpr uint32_t l1_slot_stride = get_compile_time_arg_val(9);
    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr uint32_t cb_id = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t BATCH = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);

    const auto s = TensorAccessor(src_args, src_addr, input_page_size);

    uint32_t page = out_start_page;
    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_reserve_back(cb_id, batch);
        uint32_t l1 = get_write_ptr(cb_id);

        // Read all sticks in the batch into the tail of their slots first.
        uint32_t rd = l1;
        for (uint32_t t = 0; t < batch; t++) {
            uint32_t stage = (rd + src_min_off + input_alignment - 1) & ~(input_alignment - 1);
            noc_async_read(s.get_noc_addr(page + t), stage, in_read_size);
            rd += l1_slot_stride;
        }
        noc_async_read_barrier();

        // Expand each stick front-to-back, in place.
        uint32_t slot = l1;
        for (uint32_t t = 0; t < batch; t++) {
            if constexpr (ELEM_SIZE == 2) {
                volatile tt_l1_ptr uint16_t* b = (volatile tt_l1_ptr uint16_t*)slot;
                uint32_t stage = (slot + src_min_off + input_alignment - 1) & ~(input_alignment - 1);
                volatile tt_l1_ptr uint16_t* src = (volatile tt_l1_ptr uint16_t*)stage;
                uint32_t o = 0;
                for (uint32_t i = 0; i < W_in; i++) {
                    uint16_t v = src[i];
                    for (uint32_t r = 0; r < REPEATS; r++) {
                        b[o++] = v;
                    }
                }
            } else {  // ELEM_SIZE == 4
                volatile tt_l1_ptr uint32_t* b = (volatile tt_l1_ptr uint32_t*)slot;
                uint32_t stage = (slot + src_min_off + input_alignment - 1) & ~(input_alignment - 1);
                volatile tt_l1_ptr uint32_t* src = (volatile tt_l1_ptr uint32_t*)stage;
                uint32_t o = 0;
                for (uint32_t i = 0; i < W_in; i++) {
                    uint32_t v = src[i];
                    for (uint32_t r = 0; r < REPEATS; r++) {
                        b[o++] = v;
                    }
                }
            }
            slot += l1_slot_stride;
        }

        cb_push_back(cb_id, batch);
        page += batch;
        pages_left -= batch;
    }
}
