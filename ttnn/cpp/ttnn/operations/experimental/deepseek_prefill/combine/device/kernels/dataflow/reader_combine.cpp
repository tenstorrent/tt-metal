// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

// Signal last element to writer to break out of loop
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-8)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(5);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(6);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(7);
    constexpr uint32_t output_pages = get_compile_time_arg_val(8);

    // Page sizes (indices 9-12)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(12);

    // Operation parameters (indices 13-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(13);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(14);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(15);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // Mesh information (indices 22-26)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(22);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(23);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(25);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(26);

    // Fabric configuration (indices 27-30)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(27);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(28);
    constexpr uint32_t num_links = get_compile_time_arg_val(29);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(30);

    // Batch configuration (index 31)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(31);
    // Number of dispatch groups (index 32)
    constexpr uint32_t num_dispatch_groups = get_compile_time_arg_val(32);

    // Expert region offsets tensor metadata (indices 33-36)
    constexpr uint32_t cb_expert_region_offsets_id = get_compile_time_arg_val(33);
    constexpr uint32_t expert_region_offsets_pages = get_compile_time_arg_val(34);
    constexpr uint32_t expert_region_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_expert_region_offsets_page_size = get_compile_time_arg_val(36);

    // Dispatch buffer total per-chip capacity (index 37) — used as overflow guard.
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(37);

    // TensorAccessorArgs for all 5 tensors (starting at index 38)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<38>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();
    constexpr auto expert_region_offsets_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

#if INIT_ZEROS
    // Zero-init args follow immediately after the TensorAccessorArgs block
    constexpr uint32_t cb_zero_buffer_id =
        get_compile_time_arg_val(expert_region_offsets_args.next_compile_time_args_offset());
    constexpr uint32_t num_total_untilizer_cores =
        get_compile_time_arg_val(expert_region_offsets_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t tile_layout_args_base = expert_region_offsets_args.next_compile_time_args_offset() + 2;
#else
    constexpr uint32_t tile_layout_args_base = expert_region_offsets_args.next_compile_time_args_offset();
#endif

    // Sender always consumes untilized rows + routing metadata from its dedicated untilizer
    // group's receive_buf (c_18) / metadata ring (c_19); these args are appended per-sender by
    // the program factory for both TILE_LAYOUT and ROW_MAJOR (the ROW_MAJOR untilizer reader
    // page-copies rows into c_2 instead of untilizing, but the sender path is identical).
    constexpr uint32_t num_untilizer_cores_group = get_compile_time_arg_val(tile_layout_args_base);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(tile_layout_args_base + 1);
    constexpr uint32_t cb_metadata_buf_id = get_compile_time_arg_val(tile_layout_args_base + 2);
    // Per-untilizer ring depth on the sender's receive_buf (drives the slot ring below).
    constexpr uint32_t SLOTS_PER_UNTILIZER = get_compile_time_arg_val(tile_layout_args_base + 3);

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_region_offsets_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_init_complete_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_init_complete_semaphore_address = get_semaphore(output_init_complete_semaphore_id);
    uint32_t output_init_barrier_address = get_semaphore(output_init_barrier_semaphore_id);

    DPRINT_COMBINE(
        "Combine Reader: experts=[{}, {}) linearized_mesh_coord={}\n",
        expert_start_idx,
        expert_end_idx,
        linearized_mesh_coord);

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

#if INIT_ZEROS
    // Hybrid row output-zeroing: this core zeroes its assigned page range, then waits for untilizer row cores
    {
        uint32_t page_start = get_arg_val<uint32_t>(rt_args++);
        uint32_t page_end = get_arg_val<uint32_t>(rt_args++);
        uint32_t output_init_done_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        uint32_t output_init_done_sem_address = get_semaphore(output_init_done_semaphore_id);

        {
            // DeviceZoneScopedN("combine-output-zeroing-SENDER-writing");
            zero_pages(cb_zero_buffer_id, page_start, page_end, aligned_output_page_size, output_addr_gen);
        }

        volatile tt_l1_ptr uint32_t* output_init_done_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_init_done_sem_address);
        noc_semaphore_wait(output_init_done_sem_ptr, num_total_untilizer_cores);
        noc_semaphore_set(output_init_done_sem_ptr, 0);
    }
#endif

    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t untilizer_counter_l1_offset = get_write_ptr(cb_dispatched_metadata_id);
    uint32_t counter_ready_sem_l1_offset = get_semaphore(counter_ready_semaphore_id);
    uint64_t mcast_counter_noc_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, untilizer_counter_l1_offset);
    uint64_t mcast_counter_sem_noc_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, counter_ready_sem_l1_offset);

    // Per-untilizer semaphores (each scoped to just the (this sender, untilizer) pair):
    //   data_ready: untilizer ++ after each non-local row it writes into receive_buf.  We do
    //               wait(>=1) + atomic dec(-1) to consume exactly one per row.
    //   credits:    init SLOTS_PER_UNTILIZER on untilizer's L1; we ++ untilizer's copy each time we free
    //               a row slot in its 16-deep ring on our receive_buf.
    // Both the wait-side L1 ptr (on this sender) and the inc-side NOC address (on untilizer)
    // refer to the same logical sem; pair-scoped allocation guarantees the L1 offset is
    // identical on both cores.
    volatile tt_l1_ptr uint32_t* data_ready_sem_ptrs[num_untilizer_cores_group];
    uint64_t self_data_ready_noc_addrs[num_untilizer_cores_group];
    uint64_t untilizer_credits_noc_addrs[num_untilizer_cores_group];
    uint32_t untilizer_noc_x[num_untilizer_cores_group];
    uint32_t untilizer_noc_y[num_untilizer_cores_group];
    for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
        uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        uint32_t credits_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        untilizer_noc_x[c] = get_arg_val<uint32_t>(rt_args++);
        untilizer_noc_y[c] = get_arg_val<uint32_t>(rt_args++);
        uint32_t data_ready_l1 = get_semaphore(data_ready_semaphore_id);
        uint32_t credits_l1 = get_semaphore(credits_semaphore_id);
        data_ready_sem_ptrs[c] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_l1);
        self_data_ready_noc_addrs[c] = get_noc_addr(my_x[noc_index], my_y[noc_index], data_ready_l1);
        untilizer_credits_noc_addrs[c] = get_noc_addr(untilizer_noc_x[c], untilizer_noc_y[c], credits_l1);
    }

#if INIT_ZEROS
    // Signal writer that output-zeroing is complete
    volatile tt_l1_ptr uint32_t* output_init_complete_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_init_complete_semaphore_address);
    noc_semaphore_set(output_init_complete_sem_ptr, 1);

    // Wait for ALL writers (all cores) to complete init exchange.
    // Each writer signals all readers' barrier sems via noc_semaphore_inc,
    // so this reader waits for num_cores signals before proceeding.
    volatile tt_l1_ptr uint32_t* barrier_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_init_barrier_address);
    noc_semaphore_wait(barrier_sem_ptr, num_cores);
    noc_semaphore_set(barrier_sem_ptr, 0);
#endif

    // Read expert token counts
    const auto experts_tok_counter_addr_gen = TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr);
    cb_reserve_back(cb_experts_tok_counter_id, experts_tok_counter_pages);
    uint32_t counter_base_addr = get_write_ptr(cb_experts_tok_counter_id);
    {
        // DeviceZoneScopedN("combine-reading-expert-token-counts");
        for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
            noc_async_read_page(
                i, experts_tok_counter_addr_gen, counter_base_addr + i * aligned_experts_tok_counter_page_size);
        }
        noc_async_read_barrier();
    }

    // Expert token counts: flat [num_routed_experts] array per device.
    // Decompose linearized_mesh_coord into (row, col) using physical mesh dims,
    // then map col -> dispatch_group_idx via modulo num_dispatch_groups.
    // This handles DP replicas (ndg < mesh_cols) where multiple columns share the same group.
    constexpr uint32_t mesh_row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t mesh_col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t dispatch_group_idx = mesh_col % num_dispatch_groups;
    constexpr uint32_t experts_per_dispatch_group = experts_per_chip * num_chips;
    constexpr uint32_t offset = dispatch_group_idx * experts_per_dispatch_group + mesh_row * experts_per_chip;
    // Multicast expert token counts + receive_buf_addr to all untilizer cores
    // Each sender multicasts token counts + its own receive_buf_addr to its dedicated untilizer
    // group. The mcast destination covers only this sender's k_s untilizer cores (per-sender
    // bounding box), so all senders can multicast in parallel.
    // Trailer layout (one l1_alignment region after counter_total_size bytes):
    //   [0]: receive_buf_addr  — sender's c_18 L1 offset (where untilizer NOC-writes untilized data)
    //   [1]: metadata_buf_addr — sender's c_19 L1 offset (where untilizer NOC-writes routing metadata)
    {
        // DeviceZoneScopedN("combine-sender-multicast-sending");
        constexpr uint32_t counter_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;

        volatile tt_l1_ptr uint32_t* trailer_slot =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_base_addr + counter_total_size);
        trailer_slot[0] = get_write_ptr(cb_untilize_id);
        trailer_slot[1] = get_write_ptr(cb_metadata_buf_id);

        constexpr uint32_t mcast_total_size = counter_total_size + l1_alignment;
        uint32_t off = 0;
        while (off < mcast_total_size) {
            uint32_t chunk = ((mcast_total_size - off) > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE
                                                                                       : (mcast_total_size - off);
            noc_async_write_multicast(
                counter_base_addr + off, mcast_counter_noc_addr + off, chunk, num_untilizer_cores_group);
            off += chunk;
        }
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(mcast_counter_sem_noc_addr, 1, num_untilizer_cores_group);
    }

    uint32_t untilize_base = get_write_ptr(cb_untilize_id);
    uint32_t metadata_buf_base = get_write_ptr(cb_metadata_buf_id);
    // Both receive_buf (c_18) and metadata_ring (c_19) are partitioned k_s ways:
    // untilizer c owns a SLOTS_PER_UNTILIZER-deep ring starting at
    //   untilize_base    + c * SLOTS_PER_UNTILIZER * aligned_output_page_size
    //   metadata_buf_base + c * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size
    // read_slots[c] tracks the next slot index (mod SLOTS_PER_UNTILIZER) to pull from for untilizer c.
    uint32_t read_slots[num_untilizer_cores_group];
    for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
        read_slots[c] = 0;
    }

    // Round-robin polling loop — sender polls all untilizer core CBs without blocking on any
    // single one.  Each untilizer core writes routing metadata + row data for every non-local row,
    // then sends ROUTE_INFO_SENTINEL when all its batches are complete.  Sender exits when
    // every untilizer core has signalled done, eliminating head-of-line blocking between cores.
    {
        static_assert(
            (SLOTS_PER_UNTILIZER & (SLOTS_PER_UNTILIZER - 1)) == 0, "SLOTS_PER_UNTILIZER must be a power of 2");
        constexpr uint32_t SLOTS_PER_UNTILIZER_MASK = SLOTS_PER_UNTILIZER - 1;

        uint32_t untilizer_done_count = 0;
        bool untilizer_finished[num_untilizer_cores_group];
        // consumed[c] tracks how many data_ready increments we've processed for untilizer core c.
        // The untilizer core only ever INCREMENTS data_ready_sem; the sender never decrements it.
        // Replaces the per-row noc_semaphore_inc(-1) + noc_async_atomic_barrier round-trip
        // with a local register-resident counter compare.
        uint32_t consumed[num_untilizer_cores_group];
        uint32_t ring_meta_addr[num_untilizer_cores_group][SLOTS_PER_UNTILIZER];
        uint64_t buffer_scratch_noc_addr_table[num_untilizer_cores_group][SLOTS_PER_UNTILIZER];
        for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
            untilizer_finished[c] = false;
            consumed[c] = 0;
            uint32_t meta_addr = metadata_buf_base + c * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size;
            uint32_t out_addr = untilize_base + c * SLOTS_PER_UNTILIZER * aligned_output_page_size;
            for (uint32_t s = 0; s < SLOTS_PER_UNTILIZER; s++) {
                ring_meta_addr[c][s] = meta_addr;
                buffer_scratch_noc_addr_table[c][s] = get_noc_addr(out_addr);
                meta_addr += aligned_dispatched_metadata_page_size;
                out_addr += aligned_output_page_size;
            }
        }

        while (untilizer_done_count < num_untilizer_cores_group) {
            for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
                if (untilizer_finished[c]) {
                    continue;
                }

                // Non-blocking check: data_ready lives in sender L1.  Invalidate L1 cache so the
                // load picks up any NoC-written increments from the untilizer core (the prior atomic
                // barrier used to do this for us; now we do it explicitly).
                invalidate_l1_cache();
                if (*data_ready_sem_ptrs[c] == consumed[c]) {
                    continue;
                }
                consumed[c]++;

                uint32_t slot = read_slots[c];
                volatile tt_l1_ptr uint32_t* ring_meta =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_meta_addr[c][slot]);
                uint32_t meta0 = ring_meta[0];
                uint32_t meta1 = ring_meta[1];
                uint32_t meta2 = ring_meta[2];
                uint64_t buffer_scratch_noc_addr = buffer_scratch_noc_addr_table[c][slot];
                read_slots[c] = (slot + 1) & SLOTS_PER_UNTILIZER_MASK;

                if (meta0 == ROUTE_INFO_SENTINEL) {
                    // Reset the sem so a subsequent kernel invocation starts at 0 even if the
                    // framework doesn't reset program-level sems between runs.  Pairs with
                    // consumed[c] being a local that resets at kernel entry.
                    noc_semaphore_set(data_ready_sem_ptrs[c], 0);
                    noc_async_atomic_barrier();
                    untilizer_finished[c] = true;
                    untilizer_done_count++;
                    continue;
                }

                uint32_t dst_chip = meta0;
                uint32_t output_page_idx = meta1 * num_experts_per_tok + meta2;

                if constexpr (is_1d_topology<topology>()) {
                    uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                    uint32_t distance =
                        manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);

                    cb_reserve_back(cb_route_info_id, 1);
                    uint32_t cb_base = get_write_ptr(cb_route_info_id);
                    volatile tt_l1_ptr uint32_t* route_info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base);
                    route_info[0] = route;
                    route_info[1] = distance;
                    route_info[2] = output_page_idx;
                    // FABRIC_2D: writer recomputes the EDM direction from route_info[3] (dst_chip)
                    // and ignores slots [0..1]. All four slots are written unconditionally so the
                    // 2D writer doesn't see uninitialized garbage in the dst_chip slot.
                    route_info[3] = dst_chip;
                    {
                        // DeviceZoneScopedN("sending-for-FABRIC-write");
                        uint32_t output_dst = cb_base + l1_alignment;
                        noc_async_read(buffer_scratch_noc_addr, output_dst, aligned_output_page_size);
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_route_info_id, 1);
                }
                noc_semaphore_inc<true>(untilizer_credits_noc_addrs[c], 1);
            }
        }
    }

    // Push sentinel to signal writer that all dispatches are done
    cb_reserve_back(cb_route_info_id, 1);
    volatile tt_l1_ptr uint32_t* route_info =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
    route_info[0] = ROUTE_INFO_SENTINEL;
    route_info[1] = 0;
    route_info[2] = 0;
    route_info[3] = 0;
    cb_push_back(cb_route_info_id, 1);
}
