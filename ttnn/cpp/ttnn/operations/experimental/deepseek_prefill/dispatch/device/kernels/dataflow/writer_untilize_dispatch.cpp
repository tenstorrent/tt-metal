// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Writer kernel for the untilize cores in the tile-layout dispatch path.
// Runs on RISCV_0 (data movement) of each untilize core, opposite the reader
// RISC on the same core: the reader builds the next batch's route plan while
// this kernel drains the current one and performs the NOC writes.
//
// Token batches are distributed round-robin across total_workers untilize cores:
// core i processes batches i, i+total_workers, …  Each untilize core is bound to
// ONE sender core (fabric-only) that forwards its cross-device tokens.
//
// Startup handshake:
//   Wait for the owning sender to multicast its c_4/c_5/c_6 receive-buffer L1
//   base addresses (addr_ready_semaphore), then read them from the cross-addr
//   mailbox.  c_4 = route_info slots, c_5 = payload slots, c_6 = metadata slots.
//
// For each assigned batch:
//   1. Wait for compute to finish untilizing this batch (cb_wait_front on
//      cb_untilize_id).
//   2. Wait for the reader RISC to publish this batch's route plan (cb_wait_front
//      on cb_plan_id), then read it as PlanHeader + PlanEntry[] (layout shared
//      with the reader via dispatch_plan.hpp).
//   3. For each plan entry:
//        - Local (PLAN_FLAG_LOCAL): NOC-write the token payload and its metadata
//          straight to local DRAM, bypassing the sender.  Sources are unique per
//          token / ring-rotated, so a single flush at batch end covers reuse.
//        - Cross-device: wait for a per-entry credit (space_avail; the sender
//          frees one slot per fabric send), then write route_info, payload and
//          metadata into the sender's c_4/c_5/c_6 slot (TRID-tagged, barriered
//          per entry) and bump the sender's data_avail semaphore so it forwards
//          the token over the fabric.
//   4. Flush outstanding local writes, then release the plan and untilize CBs
//      (cb_pop_front).
//
// After the last batch: drain the reader's end-of-plan sentinel page, send
// ROUTE_INFO_SENTINEL to the sender (so it stops forwarding), and full-barrier
// before exit.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/dispatch_plan.hpp"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;
constexpr uint32_t TRID_NON_LOCAL_WRITE = 1;

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(0);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t total_batches = get_compile_time_arg_val(3);
    constexpr uint32_t core_id = get_compile_time_arg_val(4);
    constexpr uint32_t total_workers = get_compile_time_arg_val(5);

    constexpr uint32_t cb_metadata_scratch_id = get_compile_time_arg_val(6);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t cb_plan_id = get_compile_time_arg_val(8);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(9);

    // Per-entry CB protocol on sender side (sender writer CBs, writer_cb_size slots deep).
    constexpr uint32_t route_info_slot_stride = get_compile_time_arg_val(10);    // l1_alignment
    constexpr uint32_t writer_cb_size = get_compile_time_arg_val(11);            // = read_batch_size = 32
    constexpr uint32_t cb_route_info_scratch_id = get_compile_time_arg_val(12);  // 16B local scratch
    constexpr uint32_t meta_scratch_slots = get_compile_time_arg_val(13);

    constexpr auto output_args = TensorAccessorArgs<14>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    // ===== Runtime args =====
    uint32_t rt_idx = 0;
    uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t addr_ready_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t cross_addr_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t data_avail_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t space_avail_semaphore_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_idx++);

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);

    // ===== Startup handshake: receive the owning sender's receive-buffer base addresses =====
    // The sender owns three L1 receive buffers this untilizer fabric-feeds — c_4 (route_info),
    // c_5 (payload), c_6 (metadata) — whose L1 base addresses are only known at runtime.  The
    // sender packs all three into our cross_addr mailbox slot, then increments addr_ready.  The
    // sender barriers those address writes before the inc, so once addr_ready fires the packed
    // addresses are already in our local L1 and the read below is safe.  We reset addr_ready to 0
    // so the slot is clean for the next program launch.
    volatile tt_l1_ptr uint32_t* addr_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(addr_ready_semaphore_id));
    noc_semaphore_wait(addr_ready_sem_ptr, 1);
    noc_semaphore_set(addr_ready_sem_ptr, 0);

    // All three base addresses arrive packed in the single mailbox slot (words [0],[1],[2]).
    volatile tt_l1_ptr uint32_t* cross_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(cross_addr_semaphore_id));
    uint32_t sender_c4_l1_addr = cross_addr_ptr[0];
    uint32_t sender_c5_l1_addr = cross_addr_ptr[1];
    uint32_t sender_c6_l1_addr = cross_addr_ptr[2];

    // Pre-compute NOC addresses for the sender's three receive buffers and its data_avail
    // semaphore.  A cross-device entry writes route_info/payload/metadata into these slots,
    // then bumps data_avail to tell the sender one slot is ready to forward over the fabric.
    uint64_t sender_c4_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c4_l1_addr);
    uint64_t sender_c5_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c5_l1_addr);
    uint64_t sender_c6_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c6_l1_addr);
    uint64_t sender_data_avail_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_avail_semaphore_id));

    // space_avail lives in our local L1; the sender increments it remotely once per slot it has
    // finished forwarding.  We poll it as a per-entry credit (wait for produced_count+1) before
    // overwriting a slot, so the sender's in-flight fabric send is never clobbered.
    volatile tt_l1_ptr uint32_t* space_avail_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(space_avail_semaphore_id));

    // Metadata scratch layout: [meta_scratch_slots local ring slots][1 trailing cross-device slot].
    // Local entries rotate through the ring so consecutive DRAM writes overlap; cross-device entries
    // stage into the single trailing slot (xdev_metadata_scratch_addr) before the NOC write to c_6.
    // route_info_scratch is a tiny local buffer where the 4×u32 route_info word group is assembled
    // before being pushed to the sender's c_4 in one NOC write.
    uint32_t metadata_scratch_addr = get_write_ptr(cb_metadata_scratch_id);
    uint32_t xdev_metadata_scratch_addr = metadata_scratch_addr + meta_scratch_slots * aligned_metadata_page_size;
    uint32_t route_info_scratch_addr = get_write_ptr(cb_route_info_scratch_id);
    volatile tt_l1_ptr uint32_t* route_info_scratch =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_info_scratch_addr);
    uint32_t produced_count = 0;
    uint32_t local_count = 0;

    DPRINT_DISPATCH(
        "Writer untilize: handshake done c4={} c5={} c6={}\n", sender_c4_l1_addr, sender_c5_l1_addr, sender_c6_l1_addr);

    // ===== Per-batch loop — drains the route plan published by the reader RISC =====
    for (uint32_t batch_idx = core_id; batch_idx < total_batches; batch_idx += total_workers) {
        // Wait for compute to finish untilizing this batch
        cb_wait_front(cb_untilize_id, read_batch_size);

        uint32_t untilize_read_ptr = get_read_ptr(cb_untilize_id);

        // Wait for reader to publish the per-batch route plan
        cb_wait_front(cb_plan_id, 1);
        uint32_t plan_addr = get_read_ptr(cb_plan_id);
        volatile tt_l1_ptr PlanHeader* plan = reinterpret_cast<volatile tt_l1_ptr PlanHeader*>(plan_addr);
        volatile tt_l1_ptr PlanEntry* entries =
            reinterpret_cast<volatile tt_l1_ptr PlanEntry*>(plan_addr + sizeof(PlanHeader));
        uint32_t entry_count = plan->entry_count;
        DPRINT_DISPATCH(
            "[W c={}] b={} draining plan entries={} produced_so_far={}\n",
            (uint32_t)core_id,
            batch_idx,
            entry_count,
            produced_count);

        {
            // Drain every entry the reader recorded for this batch.  Each entry decodes to one
            // untilized token; flags selects its destination: local DRAM here, or staged into the
            // sender's receive buffers for the sender to forward over the fabric.
            for (uint32_t e = 0; e < entry_count; e++) {
                volatile tt_l1_ptr PlanEntry* entry = &entries[e];
                uint32_t flags = entry->flags;
                uint32_t token_t = entry->token_t;
                uint32_t routed_expert = entry->routed_expert;
                uint32_t page_idx = entry->page_idx;
                uint32_t token_idx = entry->token_idx;
                uint32_t weight_k = entry->weight_k;
                uint32_t k = unpack_k(weight_k);
                int16_t weight = unpack_weight(weight_k);

                // Payload source: this token's untilized row inside the compute output CB.
                uint32_t src_addr = untilize_read_ptr + token_t * aligned_output_page_size;
                bool is_local = (flags & PLAN_FLAG_LOCAL) != 0;

                if (is_local) {
                    //  Local: NOC1 write payload + metadata directly to DRAM.
                    //  No in-loop flush — payload source (src_addr) is unique per token,
                    //  and metadata source rotates through meta_scratch_slots ring slots.
                    //  Single noc_async_writes_flushed() at batch end covers reuse.
                    noc_async_write_page(page_idx, output_addr_gen, src_addr);
                    // Per-token metadata layout (5 × int32): [src chip, global token idx, top-k
                    // slot, routed expert, routing weight].  Built into the next ring slot, then
                    // written to the same DRAM page index as the payload.
                    uint32_t meta_addr =
                        metadata_scratch_addr + (local_count % meta_scratch_slots) * aligned_metadata_page_size;
                    volatile tt_l1_ptr int32_t* meta = reinterpret_cast<volatile tt_l1_ptr int32_t*>(meta_addr);
                    meta[0] = (int32_t)linearized_mesh_coord;
                    meta[1] = (int32_t)token_idx;
                    meta[2] = (int32_t)k;
                    meta[3] = (int32_t)routed_expert;
                    meta[4] = (int32_t)weight;
                    noc_async_write_page(page_idx, metadata_addr_gen, meta_addr);
                    local_count++;
                } else {
                    // Cross-device: stage this token into one sender slot as three NOC writes —
                    // route_info (c_4), payload (c_5), metadata (c_6) — then signal data_avail so
                    // the sender forwards it over the fabric.  route/distance tell the sender's
                    // fabric writer where to send.
                    uint32_t route = entry->route;
                    uint32_t distance = entry->distance;
                    uint32_t dst_chip = entry->dst_chip;

                    // Per-entry credit: wait until the sender has fabric-sent the slot we're
                    // about to overwrite (sender writer inc's space_avail once per slot freed).
                    DPRINT_DISPATCH(
                        "[W c={}] b={} WAIT space_avail>={} (have={}) for xdev entry route={}\n",
                        (uint32_t)core_id,
                        batch_idx,
                        produced_count + 1,
                        (uint32_t)(*space_avail_sem_ptr),
                        route);
                    noc_semaphore_wait_min(space_avail_sem_ptr, produced_count + 1);

                    uint32_t slot = produced_count % writer_cb_size;

                    // Build route_info (4 × u32) in local scratch, send as one NOC write.
                    // [3]=dst_chip carries the linearized dest device index for the sender's 2D
                    // fabric route (ignored under 1D, where [0]=route/[1]=distance drive the send).
                    route_info_scratch[0] = route;
                    route_info_scratch[1] = distance;
                    route_info_scratch[2] = page_idx;
                    route_info_scratch[3] = dst_chip;
                    uint64_t c4_slot = sender_c4_base_noc_addr + slot * route_info_slot_stride;
                    noc_async_write_one_packet_with_trid(
                        route_info_scratch_addr, c4_slot, route_info_slot_stride, TRID_NON_LOCAL_WRITE);

                    // Payload: chunk to NOC_MAX_BURST_SIZE packets, each tagged with TRID.
                    uint64_t c5_slot = sender_c5_base_noc_addr + slot * aligned_output_page_size;
                    uint32_t off = 0;
                    while (off < aligned_output_page_size) {
                        uint32_t chunk = (aligned_output_page_size - off > (uint32_t)NOC_MAX_BURST_SIZE)
                                             ? (uint32_t)NOC_MAX_BURST_SIZE
                                             : (aligned_output_page_size - off);
                        noc_async_write_one_packet_with_trid(
                            src_addr + off, c5_slot + off, chunk, TRID_NON_LOCAL_WRITE);
                        off += chunk;
                    }

                    // Metadata: same 5 × int32 layout as the local path, staged in the dedicated
                    // cross-device scratch slot, then written to the sender's c_6 slot.
                    volatile tt_l1_ptr int32_t* meta =
                        reinterpret_cast<volatile tt_l1_ptr int32_t*>(xdev_metadata_scratch_addr);
                    meta[0] = (int32_t)linearized_mesh_coord;
                    meta[1] = (int32_t)token_idx;
                    meta[2] = (int32_t)k;
                    meta[3] = (int32_t)routed_expert;
                    meta[4] = (int32_t)weight;
                    uint64_t c6_slot = sender_c6_base_noc_addr + slot * aligned_metadata_page_size;
                    noc_async_write_one_packet_with_trid(
                        xdev_metadata_scratch_addr, c6_slot, aligned_metadata_page_size, TRID_NON_LOCAL_WRITE);

                    // Wait only on this entry's cross-device writes — in-flight local
                    // DRAM writes are tagged-out by TRID and keep flying.
                    noc_async_write_barrier_with_trid(TRID_NON_LOCAL_WRITE);

                    noc_semaphore_inc<true>(sender_data_avail_noc_addr, 1);

                    produced_count++;
                }
            }
        }

        // Drain all local NOC writes issued during this batch before the scratch ring or
        // untilize CB get reused. Cross-device entries already barriered per-entry.
        noc_async_writes_flushed();
        local_count = 0;

        cb_pop_front(cb_plan_id, 1);
        cb_pop_front(cb_untilize_id, read_batch_size);
    }

    // Teardown: the reader pushes one extra end-of-plan sentinel page after the last batch.
    // Consume it (its entry_count is 0, so nothing was drained above for it).
    cb_wait_front(cb_plan_id, 1);
    cb_pop_front(cb_plan_id, 1);

    // Then send ROUTE_INFO_SENTINEL as one final route_info entry into the next sender slot.
    // It takes a real slot, so wait for a space_avail credit just like any data entry; the
    // sender treats this sentinel as "no more tokens from this untilizer" and stops forwarding.

    DPRINT_DISPATCH(
        "[W c={}] loop DONE; WAIT space_avail>={} (have={}) to send SENTINEL\n",
        (uint32_t)core_id,
        produced_count + 1,
        (uint32_t)(*space_avail_sem_ptr));
    noc_semaphore_wait_min(space_avail_sem_ptr, produced_count + 1);
    DPRINT_DISPATCH("[W c={}] sending SENTINEL (produced total={})\n", (uint32_t)core_id, produced_count);

    uint32_t sentinel_slot = produced_count % writer_cb_size;
    route_info_scratch[0] = ROUTE_INFO_SENTINEL;
    route_info_scratch[1] = 0;
    route_info_scratch[2] = 0;
    route_info_scratch[3] = 0;
    uint64_t sentinel_c4_slot = sender_c4_base_noc_addr + sentinel_slot * route_info_slot_stride;
    noc_async_write_one_packet_with_trid(
        route_info_scratch_addr, sentinel_c4_slot, route_info_slot_stride, TRID_NON_LOCAL_WRITE);
    noc_async_write_barrier_with_trid(TRID_NON_LOCAL_WRITE);
    noc_semaphore_inc(sender_data_avail_noc_addr, 1);

    // noc_async_full_barrier flushes everything in-flight (including the inc) before exit.
    noc_async_full_barrier();
}
