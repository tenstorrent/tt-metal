// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Untilize core RISCV_0 — data movement.
//
// Responsible for writing tokens to local DRAM or
// delegate them to the sender core if fabric write is needed.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;
constexpr uint32_t PLAN_FLAG_LOCAL = 0x1u;
constexpr uint32_t PLAN_FLAG_END = 0x80000000u;
constexpr uint32_t PLAN_ENTRY_U32S = 8;
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

    // ===== Startup: wait for sender to multicast c_4/c_5/c_6 L1 base addresses =====
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

    uint64_t sender_c4_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c4_l1_addr);
    uint64_t sender_c5_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c5_l1_addr);
    uint64_t sender_c6_base_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_c6_l1_addr);
    uint64_t sender_data_avail_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(data_avail_semaphore_id));

    volatile tt_l1_ptr uint32_t* space_avail_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(space_avail_semaphore_id));

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
        volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_addr);
        uint32_t entry_count = plan[0];
        DPRINT_DISPATCH(
            "[W c={}] b={} draining plan entries={} produced_so_far={}\n",
            (uint32_t)core_id,
            batch_idx,
            entry_count,
            produced_count);

        {
            for (uint32_t e = 0; e < entry_count; e++) {
                uint32_t base = 8 + e * PLAN_ENTRY_U32S;
                uint32_t flags = plan[base + 0];
                uint32_t token_t = plan[base + 1];
                uint32_t routed_expert = plan[base + 2];
                uint32_t page_idx = plan[base + 3];
                uint32_t token_idx = plan[base + 4];
                uint32_t kw = plan[base + 5];
                uint32_t k = kw >> 16;
                int16_t weight = (int16_t)(kw & 0xFFFF);

                uint32_t src_addr = untilize_read_ptr + token_t * aligned_output_page_size;
                bool is_local = (flags & PLAN_FLAG_LOCAL) != 0;

                if (is_local) {
                    //  Local: NOC1 write payload + metadata directly to DRAM.
                    //  No in-loop flush — payload source (src_addr) is unique per token,
                    //  and metadata source rotates through meta_scratch_slots ring slots.
                    //  Single noc_async_writes_flushed() at batch end covers reuse.
                    noc_async_write_page(page_idx, output_addr_gen, src_addr);
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
                    uint32_t route = plan[base + 6];
                    uint32_t distance = plan[base + 7];

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
                    route_info_scratch[0] = route;
                    route_info_scratch[1] = distance;
                    route_info_scratch[2] = page_idx;
                    route_info_scratch[3] = 0;
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

    // After the last data entry: write ROUTE_INFO_SENTINEL as one final entry into the
    // next slot. Must wait for space_avail like any regular entry.
    cb_wait_front(cb_plan_id, 1);
    cb_pop_front(cb_plan_id, 1);

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
