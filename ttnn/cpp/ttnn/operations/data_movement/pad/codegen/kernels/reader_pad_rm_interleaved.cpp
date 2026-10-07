// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-major pad reader: data sticks come from DRAM, pad sticks and pad tails from an L1 pad-constant buffer.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

template <uint32_t num_bytes>
inline __attribute__((always_inline)) void fill_with_val(uint32_t dst, uint32_t val) {
    static_assert(num_bytes % sizeof(uint16_t) == 0, "RM pad values are 2B or 4B scalars");
    CoreLocalMem<volatile uint32_t> ptr(dst);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
    if constexpr (num_bytes % sizeof(uint32_t) != 0) {
        CoreLocalMem<volatile uint16_t> tail(dst + (num_bytes / sizeof(uint32_t)) * sizeof(uint32_t));
        *tail = static_cast<uint16_t>(val);
    }
}

// The tail may start 2B-aligned; 2B dtypes replicate the value in both halves, so a lone uint16 store is correct.
template <uint32_t num_bytes, uint32_t start_misalign>
inline __attribute__((always_inline)) void fill_tail_with_val(uint32_t dst, uint32_t val) {
    static_assert(num_bytes % sizeof(uint16_t) == 0, "RM pad values are 2B or 4B scalars");
    static_assert(start_misalign == 0 || start_misalign == 2, "RM sticks are 2B-aligned");
    constexpr uint32_t head = (start_misalign != 0 && num_bytes >= 2) ? 2u : 0u;
    if constexpr (head != 0) {
        CoreLocalMem<volatile uint16_t> hp(dst);
        *hp = static_cast<uint16_t>(val);
    }
    constexpr uint32_t body_bytes = num_bytes - head;
    CoreLocalMem<volatile uint32_t> ptr(dst + head);
    for (uint32_t i = 0; i < body_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
    if constexpr (body_bytes % sizeof(uint32_t) != 0) {
        CoreLocalMem<volatile uint16_t> tail(dst + head + (body_bytes / sizeof(uint32_t)) * sizeof(uint32_t));
        *tail = static_cast<uint16_t>(val);
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_src_stick = get_arg_val<uint32_t>(2);
    uint32_t start_h = get_arg_val<uint32_t>(3);
    uint32_t start_c = get_arg_val<uint32_t>(4);
    uint32_t start_n = get_arg_val<uint32_t>(5);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t H_padded = get_compile_time_arg_val(3);
    constexpr uint32_t C_padded = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded = get_compile_time_arg_val(5);
    constexpr uint32_t stick_size = get_compile_time_arg_val(6);
    constexpr uint32_t stick_size_out = get_compile_time_arg_val(7);
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(8);
    constexpr uint32_t back_pad_w_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t packed_pad_val = get_compile_time_arg_val(10);
    constexpr uint32_t BATCH = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pad = get_compile_time_arg_val(13);
    constexpr uint32_t front_pad_w_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t front_h = get_compile_time_arg_val(15);
    constexpr uint32_t front_c = get_compile_time_arg_val(16);
    constexpr uint32_t front_n = get_compile_time_arg_val(17);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(18);
    constexpr auto src_args = TensorAccessorArgs<19>();
    constexpr uint32_t in_read_size = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t dram_alignment = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t congruent_fast = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 2);

    // A DRAM->L1 read needs the destination congruent to the source modulo dram_alignment, so the direct path
    // is only taken when the CB slot pitch keeps every slot congruent and the data lands at slot offset 0.
    constexpr bool LEGACY_NEEDS_STAGE =
        (front_pad_w_bytes > 0) || (stick_size % dram_alignment != 0) || (back_pad_w_bytes % dram_alignment != 0);
    constexpr bool CONGRUENT_NEEDS_STAGE = (front_pad_w_bytes > 0) || (stick_size_out_aligned % dram_alignment != 0);
    constexpr bool NEEDS_STAGE = congruent_fast ? CONGRUENT_NEEDS_STAGE : LEGACY_NEEDS_STAGE;

    // A back pad that is not a NOC-safe self-read is written with RISC stores instead.
    constexpr bool LEGACY_BACK_PAD_RISC_FILL = (front_pad_w_bytes == 0) && (stick_size % dram_alignment == 0) &&
                                               (back_pad_w_bytes > 0) && (back_pad_w_bytes % dram_alignment != 0);
    constexpr bool CONGRUENT_BACK_PAD_RISC_FILL =
        !CONGRUENT_NEEDS_STAGE && (back_pad_w_bytes > 0) &&
        ((stick_size % dram_alignment != 0) || (back_pad_w_bytes % dram_alignment != 0));
    constexpr bool BACK_PAD_RISC_FILL = congruent_fast ? CONGRUENT_BACK_PAD_RISC_FILL : LEGACY_BACK_PAD_RISC_FILL;

    static_assert(
        (NEEDS_STAGE && !BACK_PAD_RISC_FILL) || (stick_size_out_aligned % dram_alignment == 0),
        "cb_out page pitch must be a dram_alignment multiple whenever a data stick is DRAM-read directly into the "
        "CB slot");

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer pad_cb(cb_pad);
    CircularBuffer stage_cb(cb_stage);

    fill_with_val<stick_size_out>(pad_cb.get_write_ptr(), packed_pad_val);
    const uint32_t pad_l1_addr = pad_cb.get_read_ptr();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    // One stick is one page, so the tile sequencer runs with a W extent of one page and no W front pad.
    auto seq = seq_pad_init(
        start_src_stick,
        0,
        start_h,
        start_c,
        start_n,
        1,
        H,
        C,
        N,
        1,
        H_padded,
        C_padded,
        N_padded,
        0,
        front_h,
        front_c,
        front_n);
    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        out_cb.reserve_back(batch);
        uint32_t l1_addr = out_cb.get_write_ptr();

        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t src_stick = seq_pad_next(seq);

            if (seq.is_data) {
                if constexpr (BACK_PAD_RISC_FILL) {
                    CoreLocalMem<uint32_t> out_mem(l1_addr);
                    noc.async_read(
                        s, out_mem, stick_size, {.page_id = src_stick, .offset_bytes = 0}, {.offset_bytes = 0});
                    fill_tail_with_val<back_pad_w_bytes, stick_size % sizeof(uint32_t)>(
                        l1_addr + stick_size, packed_pad_val);
                } else if constexpr (NEEDS_STAGE) {
                    // Pre-fill the whole slot, then memmove the real bytes in from an aligned staging read.
                    if constexpr (front_pad_w_bytes > 0 || back_pad_w_bytes > 0) {
                        CoreLocalMem<uint32_t> out_mem(l1_addr);
                        noc.async_read(
                            self_ep,
                            out_mem,
                            stick_size_out,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_l1_addr},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                    }
                    noc.async_read(
                        s, stage_cb, in_read_size, {.page_id = src_stick, .offset_bytes = 0}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    memmove(
                        reinterpret_cast<void*>(l1_addr + front_pad_w_bytes),
                        reinterpret_cast<void*>(stage_cb.get_read_ptr()),
                        static_cast<size_t>(stick_size));
                } else {
                    CoreLocalMem<uint32_t> out_mem(l1_addr);
                    noc.async_read(
                        s, out_mem, stick_size, {.page_id = src_stick, .offset_bytes = 0}, {.offset_bytes = 0});
                    if constexpr (back_pad_w_bytes > 0) {
                        noc.async_read(
                            self_ep,
                            out_mem,
                            back_pad_w_bytes,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_l1_addr},
                            {.offset_bytes = stick_size});
                    }
                }
            } else {
                CoreLocalMem<uint32_t> out_mem(l1_addr);
                noc.async_read(
                    self_ep,
                    out_mem,
                    stick_size_out,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_l1_addr},
                    {.offset_bytes = 0});
            }

            l1_addr += stick_size_out_aligned;
        }
        noc.async_read_barrier();
        out_cb.push_back(batch);
        sticks_left -= batch;
    }
}
