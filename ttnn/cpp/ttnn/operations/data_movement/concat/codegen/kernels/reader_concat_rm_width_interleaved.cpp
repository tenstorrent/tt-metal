// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width concat of two row-major inputs: each output stick is in0's stick followed by in1's. Sticks that fill their
// page and land on a NOC-aligned offset are read straight into the output CB, batched; otherwise the stick is
// staged in scratch and copied in by RISC stores, one output stick at a time.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    uint32_t stick_id_0 = get_arg_val<uint32_t>(1);
    uint32_t stick_id_1 = get_arg_val<uint32_t>(2);
    const uint32_t src_addr_0 = get_common_arg_val<uint32_t>(0);
    const uint32_t src_addr_1 = get_common_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t IN0_STICK_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t IN1_STICK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t IN0_PAGE_SIZE = get_compile_time_arg_val(4);
    constexpr uint32_t IN1_PAGE_SIZE = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t IN1_NOC_ALIGNMENT = get_compile_time_arg_val(7);
    constexpr uint32_t BATCH = get_compile_time_arg_val(8);

    constexpr uint32_t ta_base = 9;
    constexpr auto ta0_args = TensorAccessorArgs<ta_base>();
    constexpr auto ta1_args = TensorAccessorArgs<ta0_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(ta0_args, src_addr_0, IN0_PAGE_SIZE);
    const auto s1 = TensorAccessor(ta1_args, src_addr_1, IN1_PAGE_SIZE);

    constexpr bool in0_aligned = (IN0_STICK_SIZE == IN0_PAGE_SIZE);
    constexpr bool in1_aligned = (IN1_STICK_SIZE == IN1_PAGE_SIZE);
    // in1 lands at IN0_STICK_SIZE in the output page, which must be a legal destination for in1's transport.
    constexpr bool in1_direct = in1_aligned && ((IN0_STICK_SIZE % IN1_NOC_ALIGNMENT) == 0);

    Noc noc;
    CircularBuffer input_cb(cb_in);
    CircularBuffer scratch_cb(cb_scratch);

    if constexpr (in0_aligned && in1_direct) {
        while (num_sticks > 0) {
            const uint32_t batch = num_sticks < BATCH ? num_sticks : BATCH;
            // Slots are offsets from one write pointer, so the CB depth must be a multiple of BATCH.
            input_cb.reserve_back(batch);
            for (uint32_t stick = 0; stick < batch; ++stick) {
                const uint32_t slot = stick * OUT_PAGE_SIZE;
                noc.async_read(s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0 + stick}, {.offset_bytes = slot});
                noc.async_read(
                    s1,
                    input_cb,
                    IN1_STICK_SIZE,
                    {.page_id = stick_id_1 + stick},
                    {.offset_bytes = slot + IN0_STICK_SIZE});
            }
            noc.async_read_barrier();
            input_cb.push_back(batch);
            stick_id_0 += batch;
            stick_id_1 += batch;
            num_sticks -= batch;
        }
        return;
    }

    // Private staging: only this RISC touches it, so there is nothing to reserve or push.
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();

    for (uint32_t i = 0; i < num_sticks; i++) {
        input_cb.reserve_back(1);
        uint32_t l1_addr = input_cb.get_write_ptr();

        if constexpr (in0_aligned) {
            noc.async_read(s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
            noc.async_read_barrier();
        } else {
            noc.async_read(s0, scratch_cb, IN0_PAGE_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            CoreLocalMem<volatile uint16_t> src_ptr(scratch_addr);
            CoreLocalMem<volatile uint16_t> dst_ptr(l1_addr);
            for (uint32_t w = 0; w < IN0_STICK_SIZE / 2; ++w) {
                dst_ptr[w] = src_ptr[w];
            }
        }

        if constexpr (in1_direct) {
            noc.async_read(s1, input_cb, IN1_STICK_SIZE, {.page_id = stick_id_1}, {.offset_bytes = IN0_STICK_SIZE});
            noc.async_read_barrier();
        } else {
            noc.async_read(s1, scratch_cb, IN1_PAGE_SIZE, {.page_id = stick_id_1}, {.offset_bytes = 0});
            noc.async_read_barrier();
            CoreLocalMem<volatile uint16_t> src_ptr(scratch_addr);
            CoreLocalMem<volatile uint16_t> dst_ptr(l1_addr + IN0_STICK_SIZE);
            for (uint32_t w = 0; w < IN1_STICK_SIZE / 2; ++w) {
                dst_ptr[w] = src_ptr[w];
            }
        }

        // Staged-copy drain: RISC stores can retire after push_back, so the consumer could read the page before
        // they land. A blocking read-back of the last store (matched by its start address on Wormhole) waits for them.
        constexpr uint32_t staged_end = in1_direct ? IN0_STICK_SIZE : IN0_STICK_SIZE + IN1_STICK_SIZE;
        (void)ckernel::load_blocking(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr + ((staged_end - 1) & ~1U)));

        input_cb.push_back(1);

        stick_id_0++;
        stick_id_1++;
    }
}
