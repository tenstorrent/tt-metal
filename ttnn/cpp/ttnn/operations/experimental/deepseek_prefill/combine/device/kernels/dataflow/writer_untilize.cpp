// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of an interleaved
// DRAM output tensor, then signals the combine reader cores via semaphore.
// Deployed on untilizer cores to speed up output-zeroing process by spreading out across more banks.
//
// In TILE_LAYOUT, after output-zeroing completes this kernel also owns the per-row routing
// decision.  It walks the same expert/batch iteration as reader_untilize and the compute
// kernel (driven off the multicasted per-expert token counts in c_1), so no per-batch
// signal CB is needed.  For each batch it waits for compute to push a batch onto
// cb_untilize_id and for reader_untilize to land batch_count metadata pages on
// cb_metadata_batch_id, then walks the batch's metadata one row at a time.  Local rows
// (dst_chip == this chip) are written straight to the output tensor in DRAM with no
// sender involvement.  Non-local rows run a credit-based per-row handshake against the
// sender's receive_buf: we hold a k_s-way slice of receive_buf (SLOTS_PER_UNTILIZER deep),
// consume one credit, write the row to the next slot in our ring, barrier, then inc
// data_ready so the sender can read and fabric-forward that row.  The sender ++
// credits_sem on this core when it frees a slot.  After the expert/batch loop the
// kernel writes a ROUTE_INFO_SENTINEL into the sender's metadata ring to signal
// per-untilizer-core completion.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
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
    // The output-zeroing phase consumes additional runtime args (page range, output-init-done sem, sender NOC
    // coords), but they are only present when INIT_ZEROS=1 — the program factory omits them
    // when init_zeros=False so the kernel can still run for TILE_LAYOUT's send-loop role.
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);

    Noc noc;

#if INIT_ZEROS
    uint32_t page_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_end = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_init_done_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);

    // The semaphore was created on all worker cores (including this one),
    // so get_semaphore gives the correct L1 offset for any core with this ID.
    uint32_t output_init_done_sem_l1_offset = get_semaphore(output_init_done_semaphore_id);

    // Read sender core NOC coordinates for semaphore signaling
    uint64_t sender_sem_noc_addrs[num_sender_cores];
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        uint32_t noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        sender_sem_noc_addrs[c] = get_noc_addr(noc_x, noc_y, output_init_done_sem_l1_offset);
    }
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

#if INIT_ZEROS
    zero_pages(cb_zero_buffer_id, page_start, page_end, aligned_output_page_size, output_addr_gen);

    // Signal all sender/reader cores that output-zeroing is complete
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        noc_semaphore_inc(sender_sem_noc_addrs[c], 1);
    }

    noc.async_atomic_barrier();
#endif

    // ===== Untilized-data send path (runs for both TILE_LAYOUT and ROW_MAJOR) =====
    // cb_untilize_id (c_2) is filled by the compute kernel in TILE_LAYOUT or directly by
    // reader_untilize in ROW_MAJOR; either way this kernel forwards each row to the sender.
    //
    // Compile-time args (appended after the output-zeroing TensorAccessorArgs block):
    //   +0: cb_untilize_id                        - CB into which compute pushes untilized batches
    //   +1: cb_experts_tok_counter_id             - CB c_1 multicasted by sender; sender's
    //                                               receive_buf_addr lives at the trailer
    //   +2: experts_tok_counter_pages             - number of counter pages multicasted
    //   +3: aligned_experts_tok_counter_page_size - aligned counter page size (L1 stride)
    //   +4: read_batch_size                       - number of rows per untilize batch (== tile_height)
    //   +5: cb_metadata_batch_id                  - CB this kernel pops per-batch metadata pages
    //                                               from (pushed by reader_untilize on this same core)
    //   +6: num_experts_per_tok                   - number of experts each token is routed to (for output_page_idx)
    //   +7: aligned_dispatched_metadata_page_size - aligned metadata page size (stride in cb_metadata_batch_id)
    //   +8: linearized_mesh_coord                 - this chip's linearized (row, col) in the mesh;
    //                                               metadata[t][0] == this value means the t-th row
    //                                               of the batch stays local
    //   +9: experts_per_chip                      - count of experts mapped to this chip (counter array stride)
    //  +10: counter_offset                        - uint32 offset into counter buffer for this chip
    //  +11: max_dispatch_buffer_token_size        - per-chip dispatch capacity (overflow clamp)
    //  +12: full_ct_dim                           - hidden_size / tile_width (tiles per batch, for start_page_tiled)
    //  +13: cb_counter_total_pages                - full page capacity of c_1 (counter + trailer);
    //                                              used for cb_wait_front on the multicasted CB
    //  +14: SLOTS_PER_UNTILIZER                    - per-untilizer ring depth on the sender's receive_buf
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    CircularBuffer cb_untilize(cb_untilize_id);
    constexpr uint32_t cb_experts_tok_counter_id =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);
    CircularBuffer cb_experts_tok_counter(cb_experts_tok_counter_id);
    constexpr uint32_t experts_tok_counter_pages =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t aligned_experts_tok_counter_page_size =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t cb_metadata_batch_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 5);
    CircularBuffer cb_metadata_batch(cb_metadata_batch_id);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 6);
    constexpr uint32_t aligned_dispatched_metadata_page_size =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 7);
    constexpr uint32_t linearized_mesh_coord =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 8);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 9);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 10);
    constexpr uint32_t max_dispatch_buffer_token_size =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 11);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 12);
    constexpr uint32_t cb_counter_total_pages =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 13);
    constexpr uint32_t SLOTS_PER_UNTILIZER = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 14);
    constexpr uint32_t counter_data_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;
    // read_batch_size doubles as tile_height: one tile-row of input -> read_batch_size element rows.
    constexpr uint32_t tile_height = read_batch_size;
    constexpr uint32_t tiles_per_batch = full_ct_dim;

    // Runtime args appended after the sender_sem_noc_addrs loop above:
    //   counter_ready_semaphore_id   - sem the sender increments once after its counter multicast
    //   sender_noc_x / sender_noc_y  - NOC coords of the owning sender core
    //   data_ready_semaphore_id      - sender-side sem dedicated to THIS untilizer core (one per
    //                                  (sender, untilizer) pair); this kernel ++ once per non-local
    //                                  row after the row has landed in receive_buf.  Sender
    //                                  atomically dec(-1) per row consumed.
    //   credits_semaphore_id         - local sem (init SLOTS_PER_UNTILIZER) the sender increments
    //                                  each time it frees a row slot in our ring on its
    //                                  receive_buf.  This kernel maintains a local credit
    //                                  counter; when it hits 0 we wait for credits to come
    //                                  back, atomically suck the value, and dec(-N).
    //   core_id                      - this untilizer's local index (0..k_s-1) inside the sender's
    //                                  group; used to pick our k_s-way slice of receive_buf.
    //   expert_start_idx / expert_end_idx - expert range (now [0, experts_per_chip); every group does all experts)
    //   untilizer_global_pos              - this core's position in the global interleaved untilizer
    //                                       ordering; its batches are global_pos, +G, +2G, … per expert
    //   total_untilizers                  - G, total untilizer cores across all senders (global stride)
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t credits_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t untilizer_global_pos = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t total_untilizers = get_arg_val<uint32_t>(rt_args_idx++);

    Semaphore<> credits_sem(credits_semaphore_id);
    Semaphore<> counter_ready_sem(counter_ready_semaphore_id);

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
    // counter_ready_sem_ptr is kept around for the end-of-kernel sem reset that re-arms it
    // for the next invocation.
    volatile tt_l1_ptr uint32_t* counter_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(counter_ready_semaphore_id));
    cb_experts_tok_counter.wait_front(cb_counter_total_pages);
    uint32_t counter_cb_base = cb_experts_tok_counter.get_read_ptr();
    const volatile tt_l1_ptr uint32_t* trailer =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counter_cb_base + counter_data_total_size);
    uint32_t sender_receive_buf_l1_offset = trailer[0];
    // Our k_s-way slice in the sender's receive_buf starts at core_id * SLOTS_PER_UNTILIZER rows
    // in.  All non-local row writes for this batch (and every future batch routed through
    // this (sender, untilizer) pair) cycle through slots 0..SLOTS_PER_UNTILIZER-1 within that slice.
    uint32_t our_slice_l1_offset =
        sender_receive_buf_l1_offset + core_id * SLOTS_PER_UNTILIZER * aligned_output_page_size;
    uint64_t our_slice_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, our_slice_l1_offset);

    uint32_t sender_metadata_buf_l1_offset = trailer[1];
    uint32_t our_metadata_slice_l1_offset =
        sender_metadata_buf_l1_offset + core_id * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size;
    uint64_t our_metadata_slice_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, our_metadata_slice_l1_offset);

    uint32_t local_credits = SLOTS_PER_UNTILIZER;
    uint32_t write_slot = 0;
    constexpr uint32_t TRID_NON_LOCAL_WRITE = 1;

    // Snapshot per-expert token counts from the multicasted counter buffer.  This kernel and
    // the compute kernel (untilize_combine) iterate experts/batches the same way reader_untilize
    // does, so neither needs a per-batch signal from the other — each walks the same sequence
    // and consumes its own CBs (cb_untilize_id, cb_metadata_batch_id) in lock-step via the
    // producer-consumer protocol.
    const volatile tt_l1_ptr uint32_t* counts_l1_src =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counter_cb_base) + counter_offset;
    uint32_t local_expert_counts[experts_per_chip];
    for (uint32_t e = 0; e < experts_per_chip; e++) {
        local_expert_counts[e] = counts_l1_src[e];
    }

    // Accumulate dispatch-buffer offset for experts below our range using raw counts (host
    // laid the buffer out tile-aligned; skipped experts use their raw counts here, matching
    // reader_untilize).
    uint32_t start_page_tiled = 0;
    for (uint32_t e = 0; e < expert_start_idx; e++) {
        start_page_tiled += ((local_expert_counts[e] + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    // Per-row routing per batch: local rows go straight to DRAM; non-local rows flow through
    // a credit-based ring into the sender's receive_buf.  Credit accounting (local_credits +
    // credits_sem) outlives the batch loop — slots in our ring are reused across batches.
    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t expert_tokens = local_expert_counts[local_expert];
        uint32_t start_token = (start_page_tiled / tiles_per_batch) * tile_height;
        // Mirror reader_dispatch's overflow guard so we never route past the dispatch buffer.
        if (start_token >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_token;
        }

        uint32_t actual_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        // Global round-robin: this core handles batches untilizer_global_pos, +G, +2G, … of every
        // expert (G = total_untilizers across all senders).  Must match reader_untilize / compute
        // exactly so the c_2 / c_9 producer-consumer protocol and the per-row routing stay in
        // lockstep.
        for (uint32_t batch_idx = untilizer_global_pos; batch_idx < actual_batches; batch_idx += total_untilizers) {
            uint32_t batch_token_start = batch_idx * read_batch_size;
            uint32_t batch_count = ((batch_token_start + read_batch_size) <= expert_tokens)
                                       ? read_batch_size
                                       : (expert_tokens - batch_token_start);

            // Wait for compute to finish untilizing this batch and for reader_untilize to land
            // the corresponding metadata pages.  cb_metadata_batch_id is pushed/popped in
            // fixed read_batch_size chunks (only the first batch_count entries are valid),
            // so its fifo pointers wrap cleanly even when batch_count < read_batch_size.
            cb_untilize.wait_front(read_batch_size);
            cb_metadata_batch.wait_front(read_batch_size);

            uint32_t untilize_read_ptr = cb_untilize.get_read_ptr();
            uint32_t metadata_read_ptr = cb_metadata_batch.get_read_ptr();

            for (uint32_t t = 0; t < batch_count; t++) {
                const volatile tt_l1_ptr uint32_t* metadata = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    metadata_read_ptr + t * aligned_dispatched_metadata_page_size);
                uint32_t dst_chip = metadata[0];
                uint32_t untilize_row_addr = untilize_read_ptr + t * aligned_output_page_size;

                if (dst_chip == linearized_mesh_coord) {
                    // DeviceZoneScopedN("LOCAL-row-write");
                    uint32_t dst_token_idx = metadata[1];
                    uint32_t dst_topk_indice = metadata[2];
                    uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;
                    noc.async_write(
                        cb_untilize,
                        output_addr_gen,
                        aligned_output_page_size,
                        {.offset_bytes = t * aligned_output_page_size},
                        {.page_id = output_page_idx});

                } else {
                    if (local_credits == 0) {
                        credits_sem.wait_min(1);
                        uint32_t n = *credits_sem_ptr;
                        noc_semaphore_inc(self_credits_noc_addr, (uint32_t)(-(int32_t)n));
                        noc.async_atomic_barrier();
                        local_credits += n;
                    }
                    // Write routing metadata (dst_chip, dst_token_idx, dst_topk_indice) to
                    // sender's c_19 metadata ring slot so sender can read from L1 instead of DRAM.
                    uint64_t meta_dst_addr =
                        our_metadata_slice_noc_addr + write_slot * aligned_dispatched_metadata_page_size;
                    uint32_t meta_src_addr = metadata_read_ptr + t * aligned_dispatched_metadata_page_size;
                    noc_async_write_one_packet_with_trid(meta_src_addr, meta_dst_addr, 12, TRID_NON_LOCAL_WRITE);

                    // Write untilized row data to sender's c_18 receive_buf ring slot.
                    uint64_t dst_addr = our_slice_noc_addr + write_slot * aligned_output_page_size;
                    uint32_t off = 0;
                    {
                        // DeviceZoneScopedN("FABRIC-row-write");
                        while (off < aligned_output_page_size) {
                            uint32_t chunk = (aligned_output_page_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                                 ? (uint32_t)NOC_MAX_BURST_SIZE
                                                 : (aligned_output_page_size - off);
                            noc_async_write_one_packet_with_trid(
                                untilize_row_addr + off, dst_addr + off, chunk, TRID_NON_LOCAL_WRITE);
                            off += chunk;
                        }
                        noc_async_write_barrier_with_trid(TRID_NON_LOCAL_WRITE);  // zone measures only row-data landing
                    }

                    noc_semaphore_inc<true>(sender_data_ready_noc_addr, 1);

                    write_slot = (write_slot + 1) % SLOTS_PER_UNTILIZER;
                    local_credits--;
                }
            }
            // Make sure any local writes that hit only the writes-flushed path complete before
            // we release the CBs (sender's per-row handshake already barriered the non-local
            // writes).
            noc.async_write_barrier();

            cb_metadata_batch.pop_front(read_batch_size);
            cb_untilize.pop_front(read_batch_size);
        }
        start_page_tiled += ((expert_tokens + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    // All batches processed — send job-done sentinel to sender so it knows this untilizer
    // core has finished completely.  Uses one ring slot (credit consumed, data_ready++).

    credits_sem.wait_min(SLOTS_PER_UNTILIZER - local_credits);
    uint32_t n = *credits_sem_ptr;
    noc_semaphore_inc(self_credits_noc_addr, (uint32_t)(-(int32_t)n));
    counter_ready_sem.set(0);
    noc.async_atomic_barrier();

    uint64_t done_meta_noc = our_metadata_slice_noc_addr + write_slot * aligned_dispatched_metadata_page_size;
    noc_inline_dw_write(done_meta_noc, ROUTE_INFO_SENTINEL);
    noc.async_write_barrier();
    noc_semaphore_inc(sender_data_ready_noc_addr, 1);
    noc.async_atomic_barrier();
    cb_experts_tok_counter.pop_front(cb_counter_total_pages);
}
