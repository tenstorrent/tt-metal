// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel local writer: ships this core's per-row K-element unfused
// sequence (value tiles + index tiles, raw 32-bit) into the final core's
// gathered CBs, using the 2-semaphore gather protocol (modeled on the
// reduction topk multi-core writer_local/reader_final pair):
//   - wait for the final core's receiver semaphore (VALID = buffers reserved),
//   - NoC-write both regions at this slice's offset, barrier,
//   - reset the local receiver semaphore to INVALID *before* signaling
//     completion, so the final core's next-row VALID multicast can never be
//     clobbered by a late local reset,
//   - atomically bump the final core's sender semaphore by 1.
//
// The gathered CBs are allocated on one core range spanning local + final
// cores, so their L1 address is identical everywhere; this core reads its own
// (never-advanced) write pointer to learn the final core's destination base.
//
// An empty slice (valid_length cut this core's chunk range entirely) has no
// compute output; instead this writer sends a prefilled sequence of all -inf
// values (0xFF800000) and sentinel indices (0xFFFFFFFF). All-equal -inf lanes
// are a valid sorted run in either direction, merge cleanly, and the final
// core's mark_neginf pass turns whatever index survives into the sentinel.

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    const uint32_t final_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t final_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t slice_id = get_arg_val<uint32_t>(3);
    const uint32_t is_empty = get_arg_val<uint32_t>(4);

    constexpr uint32_t values_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
    constexpr uint32_t gathered_values_cb = get_compile_time_arg_val(3);
    constexpr uint32_t gathered_indices_cb = get_compile_time_arg_val(4);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_sequence = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t sequence_bytes = tiles_per_sequence * tile_bytes;

    Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    UnicastEndpoint remote;
    CircularBuffer values_cb_obj(values_cb);
    CircularBuffer indices_cb_obj(indices_cb);
    CircularBuffer gathered_values_obj(gathered_values_cb);
    CircularBuffer gathered_indices_obj(gathered_indices_cb);

    // Address symmetry: the gathered CBs live at the same L1 address on every
    // core in their range, and this core never pushes to them, so the local
    // write pointer is the final core's buffer base.
    const uint32_t gathered_values_base = gathered_values_obj.get_write_ptr() + slice_id * sequence_bytes;
    const uint32_t gathered_indices_base = gathered_indices_obj.get_write_ptr() + slice_id * sequence_bytes;

    if (is_empty != 0) {
        CircularBuffer scratch_cb_obj(scratch_cb);
        scratch_cb_obj.reserve_back(2 * tiles_per_sequence);
        const uint32_t scratch_base = scratch_cb_obj.get_write_ptr();
        constexpr uint32_t sequence_words = sequence_bytes / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base);
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[i] = 0xFF800000u;  // exact BF16 -inf in the FP32 DST container
        }
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[sequence_words + i] = 0xFFFFFFFFu;  // sentinel index
        }

        const CoreLocalMem<uint32_t> scratch_values(scratch_base);
        const CoreLocalMem<uint32_t> scratch_indices(scratch_base + sequence_bytes);
        for (uint32_t row = 0; row < num_rows; ++row) {
            receiver_sem.wait(VALID);
            noc.async_write(
                scratch_values,
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = final_noc_x, .noc_y = final_noc_y, .addr = gathered_values_base});
            noc.async_write(
                scratch_indices,
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = final_noc_x, .noc_y = final_noc_y, .addr = gathered_indices_base});
            noc.async_write_barrier();
            receiver_sem.set(INVALID);
            sender_sem.up(noc, final_noc_x, final_noc_y, 1);
            noc.async_atomic_barrier();
        }
        return;
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        values_cb_obj.wait_front(tiles_per_sequence);
        indices_cb_obj.wait_front(tiles_per_sequence);
        receiver_sem.wait(VALID);

        noc.async_write(
            values_cb_obj,
            remote,
            sequence_bytes,
            {.offset_bytes = 0},
            {.noc_x = final_noc_x, .noc_y = final_noc_y, .addr = gathered_values_base});
        noc.async_write(
            indices_cb_obj,
            remote,
            sequence_bytes,
            {.offset_bytes = 0},
            {.noc_x = final_noc_x, .noc_y = final_noc_y, .addr = gathered_indices_base});
        // Drain both source reads before releasing the CB slots (WAR against
        // the compute producer) and before signaling arrival at the final core.
        noc.async_write_barrier();

        receiver_sem.set(INVALID);
        sender_sem.up(noc, final_noc_x, final_noc_y, 1);
        noc.async_atomic_barrier();

        values_cb_obj.pop_front(tiles_per_sequence);
        indices_cb_obj.pop_front(tiles_per_sequence);
    }
}
