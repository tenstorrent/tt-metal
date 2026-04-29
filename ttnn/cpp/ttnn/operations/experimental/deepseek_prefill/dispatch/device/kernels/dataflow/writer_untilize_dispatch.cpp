// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Writer kernel for idle (worker) cores in the dispatch op.
// Runs on the data-movement RISC opposite the reader so that the reader can
// prefetch the next batch's DRAM tiles into cb_input_id while this kernel is
// waiting for compute / the owning sender and performing the NOC write.
//
// Each idle core is permanently bound to ONE sender core (its owning sender).
// Token batches are distributed round-robin across total_workers (k_s idle
// cores + the sender itself).  Core i processes batches i, i+total_workers, …
//
// For each assigned batch:
//   1. Wait for compute to finish (cb_wait_front on cb_untilize_id).
//   2. Wait for the owning sender's "send now" signal (start_semaphore).
//      The sender also writes a route table to cb_mailbox_id before signaling,
//      so the table is ready to read immediately after the semaphore fires.
//   3. For each local route entry in the mailbox: write output data and metadata
//      directly to DRAM using NOC1, bypassing the sender entirely.
//   4. Bulk-write the full untilized batch to the sender's receive buffer so
//      the sender can forward cross-device tokens via the fabric writer.
//   5. Signal the sender that data has landed (data_ready_semaphore).
//   6. Release the untilize CB (cb_pop_front).
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(0);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t total_batches = get_compile_time_arg_val(3);
    constexpr uint32_t core_id = get_compile_time_arg_val(4);
    constexpr uint32_t total_workers = get_compile_time_arg_val(5);
    // New: direct-DRAM-write support
    constexpr uint32_t cb_mailbox_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_metadata_scratch_id = get_compile_time_arg_val(7);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(9);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(10);
    constexpr auto output_args = TensorAccessorArgs<11>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    constexpr uint32_t total_transfer_size = read_batch_size * aligned_output_page_size;

    // ===== Runtime args =====
    uint32_t rt_idx = 0;
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t start_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t addr_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t addr_value_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    // New:
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t mbox_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t mbox_scratch_addr_semaphore_id = get_arg_val<uint32_t>(rt_idx++);

    uint64_t sender_data_ready_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_ready_semaphore_id));
    volatile tt_l1_ptr uint32_t* start_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_semaphore_id));

    // ===== Startup: receive buffer address exchange (existing) =====
    volatile tt_l1_ptr uint32_t* addr_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(addr_ready_semaphore_id));
    noc_semaphore_wait(addr_ready_sem_ptr, 1);
    noc_semaphore_set(addr_ready_sem_ptr, 0);
    volatile tt_l1_ptr uint32_t* addr_value_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(addr_value_semaphore_id));
    uint32_t sender_receive_buf_l1_addr = *addr_value_ptr;
    uint64_t sender_receive_buf_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_receive_buf_l1_addr);

    // ===== Startup: NOC-write mailbox L1 address into sender's scratch slot =====
    // addr_ready also signals that the sender has multicast its rt_scratch_base into the
    // mbox_scratch_addr semaphore slot on every core.  We read it here, then use
    // noc_inline_dw_write to push our mailbox address into the sender's scratch buffer at
    // slot [core_id * 4].  noc_inline_dw_write is ordered before the subsequent
    // noc_semaphore_inc on the NOC, so the sender's local L1 read is safe after mbox_ready fires.
    uint32_t mailbox_l1_addr = get_write_ptr(cb_mailbox_id);
    volatile tt_l1_ptr uint32_t* mbox_scratch_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mbox_scratch_addr_semaphore_id));
    uint32_t sender_scratch_base = *mbox_scratch_addr_ptr;
    uint64_t sender_scratch_slot_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_scratch_base + core_id * sizeof(uint32_t));
    noc_inline_dw_write(sender_scratch_slot_noc_addr, mailbox_l1_addr);
    uint64_t sender_mbox_ready_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(mbox_ready_semaphore_id));
    noc_semaphore_inc(sender_mbox_ready_noc_addr, 1);
    noc_async_atomic_barrier();

    // ===== Addr gens for direct DRAM writes =====
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);
    uint32_t metadata_scratch_addr = get_write_ptr(cb_metadata_scratch_id);

    for (uint32_t batch_idx = core_id; batch_idx < total_batches; batch_idx += total_workers) {
        // 1. Wait for compute to finish untilizing this batch
        cb_wait_front(cb_untilize_id, read_batch_size);

        // 2. Wait for the sender's "send now" signal
        //    The sender writes the route table to cb_mailbox_id before incrementing this semaphore,
        //    so the table is guaranteed to be visible once we clear the semaphore.
        noc_semaphore_wait(start_sem_ptr, 1);
        noc_semaphore_set(start_sem_ptr, 0);

        uint32_t untilize_read_ptr = get_read_ptr(cb_untilize_id);

        // 3. Write local-expert tokens directly to DRAM (NOC1), skipping the sender.
        //    Mailbox layout: [entry_count | has_non_local_flag (u32)] [entry_0..entry_N]
        //    Each entry: token_t, page_idx, token_idx, k, routed_expert, weight (6 × u32)
        //    High bit of mbox[0]: 1 = cross-device tokens exist → must bulk-send back to sender.
        //                         0 = all local → skip bulk-send entirely.
        volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_l1_addr);
        uint32_t raw = mbox[0];
        bool has_non_local = (raw & 0x80000000u) != 0;
        uint32_t entry_count = raw & 0x7FFFFFFFu;
        for (uint32_t e = 0; e < entry_count; e++) {
            uint32_t base = 1 + e * 6;
            uint32_t token_t = mbox[base + 0];
            uint32_t page_idx = mbox[base + 1];
            int32_t token_idx = (int32_t)mbox[base + 2];
            int32_t k = (int32_t)mbox[base + 3];
            int32_t routed_exp = (int32_t)mbox[base + 4];
            int16_t weight = (int16_t)mbox[base + 5];

            uint32_t src_addr = untilize_read_ptr + token_t * aligned_output_page_size;

            noc_async_write_page(page_idx, output_addr_gen, src_addr);

            volatile tt_l1_ptr int32_t* meta = reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_scratch_addr);
            meta[0] = (int32_t)linearized_mesh_coord;
            meta[1] = token_idx;
            meta[2] = k;
            meta[3] = routed_exp;
            meta[4] = (int32_t)weight;
            noc_async_write_page(page_idx, metadata_addr_gen, metadata_scratch_addr);
            noc_async_writes_flushed();
        }

        // 4. Bulk-write full untilized batch to sender's receive buffer (cross-device tokens).
        //    Skipped entirely when all tokens are local — no round-trip to sender needed.
        if (has_non_local) {
            uint32_t off = 0;
            while (off < total_transfer_size) {
                uint32_t chunk = (total_transfer_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                     ? (uint32_t)NOC_MAX_BURST_SIZE
                                     : (total_transfer_size - off);
                noc_async_write(untilize_read_ptr + off, sender_receive_buf_noc_addr + off, chunk);
                off += chunk;
            }
            noc_async_write_barrier();

            // 5. Signal the sender that all data has landed
            noc_semaphore_inc(sender_data_ready_noc_addr, 1);
            noc_async_atomic_barrier();
        }

        // 6. Release untilize CB
        cb_pop_front(cb_untilize_id, read_batch_size);
    }
}
