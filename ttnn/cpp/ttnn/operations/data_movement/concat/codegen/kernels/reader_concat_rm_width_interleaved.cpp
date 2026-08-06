// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Concat reader for RM width-dim concat (interleaved).
// Each output stick = [in0_stick | in1_stick].
// Handles BH 64B alignment: uses aligned page sizes for TensorAccessor
// and scratch buffer for non-aligned reads.
//
// CT args layout:
//   [0]   cb_in
//   [1]   IN0_STICK_SIZE (actual bytes for in0 stick)
//   [2]   IN1_STICK_SIZE (actual bytes for in1 stick)
//   [3]   OUT_PAGE_SIZE  (aligned output page size)
//   [4]   IN0_PAGE_SIZE  (aligned page size for in0)
//   [5]   IN1_PAGE_SIZE  (aligned page size for in1)
//   [6]   cb_scratch
//   [7]   IN1_NOC_ALIGNMENT (source transport's legal destination alignment)
//   [8]   BATCH (read batch size on the aligned direct-write fast path;
//               matches the writer's BATCH so both sides drive the same
//               2*BATCH-deep CB -- see reader_concat_rm_interleaved.cpp)
//   [9..] TensorAccessorArgs for tensor 0, then tensor 1
//
// RT args layout:
//   [0]   num_sticks (total output sticks this core produces)
//   [1]   src_addr_0
//   [2]   src_addr_1
//   [3]   start_stick_0 (start stick id in tensor 0)
//   [4]   start_stick_1 (start stick id in tensor 1)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_0 = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t stick_id_0 = get_arg_val<uint32_t>(3);
    uint32_t stick_id_1 = get_arg_val<uint32_t>(4);

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

    // TensorAccessors use aligned page sizes for correct NOC address computation
    const auto s0 = TensorAccessor(ta0_args, src_addr_0, IN0_PAGE_SIZE);
    const auto s1 = TensorAccessor(ta1_args, src_addr_1, IN1_PAGE_SIZE);

    constexpr bool in0_aligned = (IN0_STICK_SIZE == IN0_PAGE_SIZE);
    constexpr bool in1_aligned = (IN1_STICK_SIZE == IN1_PAGE_SIZE);
    // Input 1 lands after input 0 inside the output page. Even when input 1's
    // source page is aligned, that destination is not a legal NOC endpoint if
    // the first payload is not aligned for input 1's source transport.  DRAM
    // therefore needs the architecture's DRAM alignment; an L1 source only
    // needs the L1 alignment. Stage through scratch and RISC-copy otherwise.
    constexpr bool in1_direct = in1_aligned && ((IN0_STICK_SIZE % IN1_NOC_ALIGNMENT) == 0);
    constexpr bool fast_path = in0_aligned && in1_direct;

    Noc noc;
    CircularBuffer input_cb(cb_in);
    CircularBuffer scratch_cb(cb_scratch);

    if constexpr (fast_path) {
        // Both sticks land directly in disjoint ranges of the same reserved CB
        // page, with no scratch involved.  Batch BATCH sticks into the CB
        // depth the factory already allocates (2*BATCH pages) and issue one
        // barrier per batch instead of one barrier per stick -- the CB was
        // deep enough to overlap read and write all along, but barriering
        // after every single stick serialized the reader against itself.
        uint32_t sticks_left = num_sticks;
        while (sticks_left > 0) {
            const uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
            input_cb.reserve_back(batch);
            for (uint32_t i = 0; i < batch; ++i) {
                const uint32_t page_offset = i * OUT_PAGE_SIZE;
                noc.async_read(
                    s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0 + i}, {.offset_bytes = page_offset});
                noc.async_read(
                    s1,
                    input_cb,
                    IN1_STICK_SIZE,
                    {.page_id = stick_id_1 + i},
                    {.offset_bytes = page_offset + IN0_STICK_SIZE});
            }
            noc.async_read_barrier();
            input_cb.push_back(batch);
            stick_id_0 += batch;
            stick_id_1 += batch;
            sticks_left -= batch;
        }
    } else {
        // Non-aligned fallback: at least one side needs a scratch-staged,
        // per-element copy, so stay stick-by-stick.
        scratch_cb.reserve_back(1);
        const uint32_t scratch_addr = scratch_cb.get_write_ptr();

        for (uint32_t i = 0; i < num_sticks; i++) {
            input_cb.reserve_back(1);
            uint32_t l1_addr = input_cb.get_write_ptr();

            // Read in0's stick
            if constexpr (in0_aligned) {
                noc.async_read(s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
                noc.async_read_barrier();
            } else {
                // Non-aligned: read aligned page into scratch, copy actual bytes
                noc.async_read(s0, scratch_cb, IN0_PAGE_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                CoreLocalMem<volatile uint16_t> src_ptr(scratch_addr);
                CoreLocalMem<volatile uint16_t> dst_ptr(l1_addr);
                for (uint32_t w = 0; w < IN0_STICK_SIZE / 2; ++w) {
                    dst_ptr[w] = src_ptr[w];
                }
            }

            // Read in1's stick right after in0's
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

            input_cb.push_back(1);

            stick_id_0++;
            stick_id_1++;
        }
    }
}
