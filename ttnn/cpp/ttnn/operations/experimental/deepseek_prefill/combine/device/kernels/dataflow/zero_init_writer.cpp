// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of an interleaved
// DRAM output tensor, then signals the combine reader cores via semaphore.
// Deployed on idle cores to speed up zero-init process by spreading out across more banks.
//
// In TILE_LAYOUT, after zero-init completes this kernel also takes over the untilized-data
// send path (previously in reader_untilize.cpp steps 3-7): it waits for compute to push a
// batch onto cb_untilize_id, waits for the owning sender's "send now" signal, NOC-writes
// the untilized rows to the sender's receive buffer, and signals the sender that data has
// landed. The loop exits when compute pushes ROUTE_INFO_SENTINEL onto cb_stop_signal_id.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"

// Sentinel used by compute to tell this kernel to exit its send loop.
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_sender_cores = get_compile_time_arg_val(1);
    constexpr uint32_t cb_zero_buffer_id = get_compile_time_arg_val(2);

    // TensorAccessorArgs for the output tensor (starting at index 3)
    constexpr auto output_args = TensorAccessorArgs<3>();

    // ===== Runtime args =====
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_end = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t zi_done_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);

    // The semaphore was created on all worker cores (including this one),
    // so get_semaphore gives the correct L1 offset for any core with this ID.
    uint32_t zi_done_sem_l1_offset = get_semaphore(zi_done_semaphore_id);

    // Read sender core NOC coordinates for semaphore signaling
    uint64_t sender_sem_noc_addrs[num_sender_cores];
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        uint32_t noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        sender_sem_noc_addrs[c] = get_noc_addr(noc_x, noc_y, zi_done_sem_l1_offset);
    }

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

    fill_zero_buffer(cb_zero_buffer_id);
    uint32_t zero_buffer_addr = get_write_ptr(cb_zero_buffer_id);

    zero_pages(zero_buffer_addr, page_start, page_end, aligned_output_page_size, output_addr_gen);

    // Signal all sender/reader cores that zero-init is complete
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        noc_semaphore_inc(sender_sem_noc_addrs[c], 1);
    }

    noc_async_atomic_barrier();

#if IS_TILE_LAYOUT
    // ===== Untilized-data send path (moved from reader_untilize.cpp steps 3-7) =====
    //
    // Compile-time args (appended after the zero-init TensorAccessorArgs block):
    //   +0: cb_untilize_id                        - CB into which compute pushes untilized batches
    //   +1: cb_stop_signal_id                     - CB for compute -> this-kernel per-batch / stop signal
    //   +2: cb_experts_tok_counter_id             - CB c_1 multicasted by sender; sender's
    //                                               receive_buf_addr lives at the trailer
    //   +3: experts_tok_counter_pages             - number of counter pages multicasted
    //   +4: aligned_experts_tok_counter_page_size - aligned counter page size (L1 stride)
    //   +5: read_batch_size                       - number of rows per untilize batch
    //   +6: cb_metadata_batch_id                  - this idle core's c_9 CB (target of sender's metadata unicast)
    //   +7: num_experts_per_tok                   - number of experts each token is routed to (for output_page_idx)
    //   +8: aligned_dispatched_metadata_page_size - aligned metadata page size (stride in c_9)
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    constexpr uint32_t cb_stop_signal_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_experts_tok_counter_id =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t experts_tok_counter_pages =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t aligned_experts_tok_counter_page_size =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 5);
    constexpr uint32_t cb_metadata_batch_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 6);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 7);
    constexpr uint32_t aligned_dispatched_metadata_page_size =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 8);
    constexpr uint32_t total_transfer_size = read_batch_size * aligned_output_page_size;
    constexpr uint32_t counter_data_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;

    // Runtime args appended after the sender_sem_noc_addrs loop above:
    //   counter_ready_semaphore_id   - sem the sender increments once after its counter multicast
    //   sender_noc_x / sender_noc_y  - NOC coords of the owning sender core
    //   data_ready_semaphore_id      - sender-side sem this kernel increments after each send
    //                                  (also used once up-front to signal the c_9 addr handshake)
    //   start_semaphore_id           - local sem the sender increments to say "send now"
    //   core_id                      - this idle core's local index within its sender's group (0..k_s-1)
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(rt_args_idx++);

    uint64_t sender_data_ready_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_ready_semaphore_id));
    volatile tt_l1_ptr uint32_t* start_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_semaphore_id));

    // Wait on the same counter_ready sem reader_untilize waits on (neither kernel resets it,
    // so both see the single increment from the sender's multicast).  Then read the sender's
    // receive_buf_addr AND its idle-c9-addr scratch L1 offset from c_1's trailer on this core.
    //   trailer[0] = sender's c_18 L1 offset (receive buffer for untilized data)
    //   trailer[1] = sender's c_10 L1 offset (scratch where this kernel writes its own c_9 addr)
    volatile tt_l1_ptr uint32_t* counter_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(counter_ready_semaphore_id));
    noc_semaphore_wait(counter_ready_sem_ptr, 1);

    uint32_t counter_cb_base = get_write_ptr(cb_experts_tok_counter_id);
    const volatile tt_l1_ptr uint32_t* trailer =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counter_cb_base + counter_data_total_size);
    uint32_t sender_receive_buf_l1_offset = trailer[0];
    uint32_t sender_idle_c9_scratch_l1_offset = trailer[1];
    uint64_t sender_receive_buf_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_receive_buf_l1_offset);

    // ===== c_9 address handshake =====
    // Write this core's c_9 L1 offset into sender's c_10 scratch at slot core_id, then increment
    // sender's data_ready_sem.  After all k_s idle cores have done this the sender sees
    // data_ready_sem == num_idle_cores_group and reads the array of addresses.  data_ready_sem
    // is reset to 0 by the sender before the batch loop, so it can be reused per-batch below.
    uint32_t my_c9_l1_offset = get_write_ptr(cb_metadata_batch_id);
    uint64_t sender_c9_slot_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_idle_c9_scratch_l1_offset + core_id * sizeof(uint32_t));
    noc_inline_dw_write(sender_c9_slot_noc_addr, my_c9_l1_offset);
    noc_async_write_barrier();  // ensure c_9 offset has landed before atomic inc wakes sender
    noc_semaphore_inc(sender_data_ready_noc_addr, 1);
    noc_async_atomic_barrier();

    // Process untilized batches until compute pushes ROUTE_INFO_SENTINEL onto cb_stop_signal_id.
    // A non-sentinel value means "another batch is on cb_untilize_id".
    // After the sender signals start_sem, c_9[0] tells us which path to take:
    //   ROUTE_INFO_SENTINEL → sender has non-local writes; send untilized data to sender
    //   any other value     → batch_count; all writes are local; write directly to output DRAM
    while (true) {
        cb_wait_front(cb_stop_signal_id, 1);
        volatile tt_l1_ptr uint32_t* stop_signal_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_stop_signal_id));
        uint32_t signal_value = stop_signal_ptr[0];
        cb_pop_front(cb_stop_signal_id, 1);
        if (signal_value == ROUTE_INFO_SENTINEL) {
            break;
        }

        // 3. Wait for compute to finish untilizing this batch
        cb_wait_front(cb_untilize_id, read_batch_size);

        // 4. Wait for the sender's "send now" signal.
        //    By the time start_sem arrives, sender has already written metadata to c_9 and
        //    set c_9[0] to ROUTE_INFO_SENTINEL (non-local) or batch_count (all-local).
        noc_semaphore_wait(start_sem_ptr, 1);
        noc_semaphore_set(start_sem_ptr, 0);

        uint32_t untilize_read_ptr = get_read_ptr(cb_untilize_id);
        volatile tt_l1_ptr uint32_t* c9_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_c9_l1_offset);

        if (c9_ptr[0] == ROUTE_INFO_SENTINEL) {
            // 5a. Non-local path: send untilized rows to the sender's receive buffer
            uint32_t off = 0;
            while (off < total_transfer_size) {
                uint32_t chunk = (total_transfer_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                     ? (uint32_t)NOC_MAX_BURST_SIZE
                                     : (total_transfer_size - off);
                noc_async_write(untilize_read_ptr + off, sender_receive_buf_noc_addr + off, chunk);
                off += chunk;
            }
            noc_async_write_barrier();
        } else {
            // 5b. All-local path: c9[0] = batch_count, write each row directly to output DRAM.
            //     Metadata layout in c_9: each entry is aligned_dispatched_metadata_page_size bytes,
            //     with [0]=dst_chip (ignored, all local), [1]=dst_token_idx, [2]=dst_topk_indice.
            uint32_t batch_count = c9_ptr[0];
            for (uint32_t t = 0; t < batch_count; t++) {
                const volatile tt_l1_ptr uint32_t* metadata = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    my_c9_l1_offset + t * aligned_dispatched_metadata_page_size);
                uint32_t dst_token_idx = metadata[1];
                uint32_t dst_topk_indice = metadata[2];
                uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;
                noc_async_write_page(
                    output_page_idx, output_addr_gen, untilize_read_ptr + t * aligned_output_page_size);
                noc_async_writes_flushed();
            }
            noc_async_write_barrier();
        }

        // 6. Signal the sender that this batch is done (data sent OR local writes done)
        noc_semaphore_inc(sender_data_ready_noc_addr, 1);
        noc_async_atomic_barrier();

        // 7. Release untilize CB
        cb_pop_front(cb_untilize_id, read_batch_size);
    }
#endif
}
