// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of an interleaved
// DRAM output tensor, then signals the combine reader cores via semaphore.
// Deployed on idle cores to speed up zero-init process by spreading out across more banks.
//
// In TILE_LAYOUT, after zero-init completes this kernel also owns the per-row routing
// decision: it waits for compute to push a batch onto cb_untilize_id, then walks the
// batch's metadata one row at a time.  Local rows (dst_chip == this chip) are written
// straight to the output tensor in DRAM with no sender involvement.  Non-local rows
// run a tight per-row handshake with the sender — wait its GO (start_sem), write this
// single row into receive_buf at the next compact slot, barrier, then inc data_ready
// so the sender can read it back and fabric-forward to its destination chip before we
// move to the next row.  The loop exits when compute pushes ROUTE_INFO_SENTINEL on
// cb_stop_signal_id.
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
    // output_addr is always needed (the all-local send-loop path writes directly to output DRAM).
    // The zero-init phase consumes additional runtime args (page range, zi-done sem, sender NOC
    // coords), but they are only present when INIT_ZEROS=1 — the program factory omits them
    // when init_zeros=False so the kernel can still run for TILE_LAYOUT's send-loop role.
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);

#if INIT_ZEROS
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
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

#if INIT_ZEROS
    fill_zero_buffer(cb_zero_buffer_id);
    uint32_t zero_buffer_addr = get_write_ptr(cb_zero_buffer_id);

    zero_pages(zero_buffer_addr, page_start, page_end, aligned_output_page_size, output_addr_gen);

    // Signal all sender/reader cores that zero-init is complete
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        noc_semaphore_inc(sender_sem_noc_addrs[c], 1);
    }

    noc_async_atomic_barrier();
#endif

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
    //   +6: cb_metadata_batch_id                  - CB this kernel pops per-batch metadata pages
    //                                               from (pushed by reader_untilize on this same
    //                                               core; sender no longer writes to it)
    //   +7: num_experts_per_tok                   - number of experts each token is routed to (for output_page_idx)
    //   +8: aligned_dispatched_metadata_page_size - aligned metadata page size (stride in cb_metadata_batch_id)
    //   +9: linearized_mesh_coord                 - this chip's linearized (row, col) in the mesh;
    //                                               metadata[t][0] == this value means the t-th row
    //                                               of the batch stays local
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
    constexpr uint32_t linearized_mesh_coord =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 9);
    constexpr uint32_t counter_data_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;

    // Runtime args appended after the sender_sem_noc_addrs loop above:
    //   counter_ready_semaphore_id   - sem the sender increments once after its counter multicast
    //   sender_noc_x / sender_noc_y  - NOC coords of the owning sender core
    //   data_ready_semaphore_id      - sender-side sem dedicated to THIS idle core (one per
    //                                  (sender, idle) pair); this kernel increments it once
    //                                  per non-local row after the row has landed in the
    //                                  sender's receive_buf.  Local rows do not touch it.
    //                                  Because every idle core has its own sem, there is no
    //                                  contention with peers in the same group.
    //   start_semaphore_id           - local sem the sender increments once per non-local row
    //                                  to hand the next compact receive_buf slot to this idle
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);

    uint64_t sender_data_ready_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_ready_semaphore_id));
    volatile tt_l1_ptr uint32_t* start_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_semaphore_id));

    // Wait on the same counter_ready sem reader_untilize waits on (neither kernel resets it,
    // so both see the single increment from the sender's multicast).  Then read the sender's
    // receive_buf_addr from c_1's trailer on this core.
    //   trailer[0] = sender's c_18 L1 offset (receive buffer for untilized data)
    volatile tt_l1_ptr uint32_t* counter_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(counter_ready_semaphore_id));
    noc_semaphore_wait(counter_ready_sem_ptr, 1);

    uint32_t counter_cb_base = get_write_ptr(cb_experts_tok_counter_id);
    const volatile tt_l1_ptr uint32_t* trailer =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counter_cb_base + counter_data_total_size);
    uint32_t sender_receive_buf_l1_offset = trailer[0];
    uint64_t sender_receive_buf_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_receive_buf_l1_offset);

    // Per-row routing: every batch goes through the same flow regardless of where the rows
    // land.  Local-destined rows are written straight to the output tensor in DRAM here.
    // Non-local-destined rows are packed contiguously into the sender's receive_buf in the
    // order they appear in the batch's metadata, one row per non-local slot.  The sender's
    // routing loop reads them back from receive_buf at the same compact offsets and forwards
    // each row to its remote chip via fabric.  start_sem hands ownership of receive_buf from
    // sender → idle for the upcoming batch; data_ready hands it back idle → sender once
    // every local DRAM write and non-local receive_buf write for the batch is in flight.
    while (true) {
        cb_wait_front(cb_stop_signal_id, 1);
        volatile tt_l1_ptr uint32_t* stop_signal_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_stop_signal_id));
        uint32_t signal_value = stop_signal_ptr[0];
        cb_pop_front(cb_stop_signal_id, 1);
        if (signal_value == ROUTE_INFO_SENTINEL) {
            break;
        }
        uint32_t batch_count = signal_value;

        // Wait for compute to finish untilizing this batch and for reader_untilize to land
        // the corresponding metadata pages in the metadata CB.
        cb_wait_front(cb_untilize_id, read_batch_size);
        cb_wait_front(cb_metadata_batch_id, batch_count);

        uint32_t untilize_read_ptr = get_read_ptr(cb_untilize_id);
        uint32_t metadata_read_ptr = get_read_ptr(cb_metadata_batch_id);

        // Per-row processing.  Local rows are written straight to the output tensor with no
        // sender involvement.  Non-local rows do a tight per-row handshake with the sender:
        // wait its GO (start_sem), write this single row into its receive_buf at the next
        // compact slot, barrier, then inc data_ready so it can read and fabric-forward this
        // row before we move to the next one.
        uint32_t nonlocal_idx = 0;
        for (uint32_t t = 0; t < batch_count; t++) {
            const volatile tt_l1_ptr uint32_t* metadata = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                metadata_read_ptr + t * aligned_dispatched_metadata_page_size);
            uint32_t dst_chip = metadata[0];
            uint32_t untilize_row_addr = untilize_read_ptr + t * aligned_output_page_size;

            if (dst_chip == linearized_mesh_coord) {
                uint32_t dst_token_idx = metadata[1];
                uint32_t dst_topk_indice = metadata[2];
                uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;
                noc_async_write_page(output_page_idx, output_addr_gen, untilize_row_addr);
                noc_async_writes_flushed();
            } else {
                noc_semaphore_wait(start_sem_ptr, 1);
                noc_semaphore_set(start_sem_ptr, 0);

                uint64_t dst_addr = sender_receive_buf_noc_addr + nonlocal_idx * aligned_output_page_size;
                uint32_t off = 0;
                while (off < aligned_output_page_size) {
                    uint32_t chunk = (aligned_output_page_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                         ? (uint32_t)NOC_MAX_BURST_SIZE
                                         : (aligned_output_page_size - off);
                    noc_async_write(untilize_row_addr + off, dst_addr + off, chunk);
                    off += chunk;
                }
                noc_async_write_barrier();

                noc_semaphore_inc(sender_data_ready_noc_addr, 1);
                noc_async_atomic_barrier();
                nonlocal_idx++;
            }
        }
        // Make sure any local writes that hit only the writes-flushed path complete before
        // we release the CBs (sender's per-row handshake already barriered the non-local
        // writes).
        noc_async_write_barrier();

        cb_pop_front(cb_metadata_batch_id, batch_count);
        cb_pop_front(cb_untilize_id, read_batch_size);
    }
    noc_semaphore_set(counter_ready_sem_ptr, 0);
#endif
}
