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
// run a credit-based per-row handshake against the sender's receive_buf: we hold a
// k_s-way slice of receive_buf (SLOTS_PER_IDLE deep), consume one credit, write the
// row to the next slot in our ring, barrier, then inc data_ready so the sender can
// read and fabric-forward that row.  The sender ++ credits_sem on this core when it
// frees a slot.  The loop exits when compute pushes ROUTE_INFO_SENTINEL on
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
    //                                  (sender, idle) pair); this kernel ++ once per non-local
    //                                  row after the row has landed in receive_buf.  Sender
    //                                  atomically dec(-1) per row consumed.
    //   credits_semaphore_id         - local sem (init SLOTS_PER_IDLE) the sender increments
    //                                  each time it frees a row slot in our ring on its
    //                                  receive_buf.  This kernel maintains a local credit
    //                                  counter; when it hits 0 we wait for credits to come
    //                                  back, atomically suck the value, and dec(-N).
    //   core_id                      - this idle's local index (0..k_s-1) inside the sender's
    //                                  group; used to pick our k_s-way slice of receive_buf.
    constexpr uint32_t SLOTS_PER_IDLE = 16;
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t credits_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(rt_args_idx++);

    uint64_t sender_data_ready_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_ready_semaphore_id));
    uint32_t credits_sem_l1 = get_semaphore(credits_semaphore_id);
    volatile tt_l1_ptr uint32_t* credits_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credits_sem_l1);
    uint64_t self_credits_noc_addr = get_noc_addr(my_x[noc_index], my_y[noc_index], credits_sem_l1);

    // Wait on the same counter_ready sem reader_untilize waits on (neither kernel resets it,
    // so both see the single increment from the sender's multicast).  Then read the sender's
    // receive_buf_addr from c_1's trailer on this core.
    //   trailer[0] = sender's c_18 L1 offset (receive buffer for untilized data)
    //   trailer[1] = sender's c_19 L1 offset (metadata ring for routing info)
    volatile tt_l1_ptr uint32_t* counter_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(counter_ready_semaphore_id));
    noc_semaphore_wait(counter_ready_sem_ptr, 1);

    uint32_t counter_cb_base = get_write_ptr(cb_experts_tok_counter_id);
    const volatile tt_l1_ptr uint32_t* trailer =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counter_cb_base + counter_data_total_size);
    uint32_t sender_receive_buf_l1_offset = trailer[0];
    // Our k_s-way slice in the sender's receive_buf starts at core_id * SLOTS_PER_IDLE rows
    // in.  All non-local row writes for this batch (and every future batch routed through
    // this (sender, idle) pair) cycle through slots 0..SLOTS_PER_IDLE-1 within that slice.
    uint32_t our_slice_l1_offset = sender_receive_buf_l1_offset + core_id * SLOTS_PER_IDLE * aligned_output_page_size;
    uint64_t our_slice_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, our_slice_l1_offset);

    uint32_t sender_metadata_buf_l1_offset = trailer[1];
    uint32_t our_metadata_slice_l1_offset =
        sender_metadata_buf_l1_offset + core_id * SLOTS_PER_IDLE * aligned_dispatched_metadata_page_size;
    uint64_t our_metadata_slice_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, our_metadata_slice_l1_offset);

    uint32_t local_credits = SLOTS_PER_IDLE;
    uint32_t write_slot = 0;

    // Per-row routing: each batch processes its rows one at a time.  Local-destined rows are
    // written straight to the output tensor in DRAM.  Non-local-destined rows are written
    // into our k_s-way slice of the sender's receive_buf at the next write_slot position
    // (mod SLOTS_PER_IDLE).  Credit-based flow control keeps us from outrunning the sender:
    // we hold a local credit counter (init SLOTS_PER_IDLE) that drops by one per non-local
    // write, and the sender ++ our credits_sem each time it frees a slot.  data_ready on
    // the sender is ++ once per non-local row written.
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
        // sender involvement.  Non-local rows use credit-based flow control: we hold a local
        // credit counter (initialized to SLOTS_PER_IDLE), one credit per free slot in our
        // ring on the sender's receive_buf.  Before each non-local row we consume a credit
        // (waiting and atomically sucking from credits_sem if we're empty), write the row
        // into our slice at write_slot, barrier, then ++ data_ready so the sender can read
        // and fabric-forward that row.  Sender ++ credits_sem each time it consumes a row,
        // releasing the slot for reuse.
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
                if (local_credits == 0) {
                    noc_semaphore_wait_min(credits_sem_ptr, 1);
                    uint32_t n = *credits_sem_ptr;
                    noc_semaphore_inc(self_credits_noc_addr, (uint32_t)(-(int32_t)n));
                    noc_async_atomic_barrier();
                    local_credits += n;
                }

                // Write routing metadata (dst_chip, dst_token_idx, dst_topk_indice) to
                // sender's c_19 metadata ring slot so sender can read from L1 instead of DRAM.
                uint64_t meta_dst_addr =
                    our_metadata_slice_noc_addr + write_slot * aligned_dispatched_metadata_page_size;
                uint32_t meta_src_addr = metadata_read_ptr + t * aligned_dispatched_metadata_page_size;
                noc_async_write(meta_src_addr, meta_dst_addr, 12);  // 3 × uint32

                // Write untilized row data to sender's c_18 receive_buf ring slot.
                uint64_t dst_addr = our_slice_noc_addr + write_slot * aligned_output_page_size;
                uint32_t off = 0;
                while (off < aligned_output_page_size) {
                    uint32_t chunk = (aligned_output_page_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                         ? (uint32_t)NOC_MAX_BURST_SIZE
                                         : (aligned_output_page_size - off);
                    noc_async_write(untilize_row_addr + off, dst_addr + off, chunk);
                    off += chunk;
                }
                noc_async_write_barrier();  // ensures both metadata and row data have landed

                noc_semaphore_inc(sender_data_ready_noc_addr, 1);
                noc_async_atomic_barrier();

                write_slot = (write_slot + 1) % SLOTS_PER_IDLE;
                local_credits--;
            }
        }
        // Make sure any local writes that hit only the writes-flushed path complete before
        // we release the CBs (sender's per-row handshake already barriered the non-local
        // writes).
        noc_async_write_barrier();

        cb_pop_front(cb_metadata_batch_id, batch_count);
        cb_pop_front(cb_untilize_id, read_batch_size);
    }

    // All batches processed — send job-done sentinel to sender so it knows this idle
    // core has finished completely.  Uses one ring slot (credit consumed, data_ready++).
    if (local_credits == 0) {
        noc_semaphore_wait_min(credits_sem_ptr, 1);
        uint32_t n = *credits_sem_ptr;
        noc_semaphore_inc(self_credits_noc_addr, (uint32_t)(-(int32_t)n));
        noc_async_atomic_barrier();
        local_credits += n;
    }
    uint64_t done_meta_noc = our_metadata_slice_noc_addr + write_slot * aligned_dispatched_metadata_page_size;
    noc_inline_dw_write(done_meta_noc, ROUTE_INFO_SENTINEL);
    noc_async_write_barrier();
    noc_semaphore_inc(sender_data_ready_noc_addr, 1);
    noc_async_atomic_barrier();
#endif
}
