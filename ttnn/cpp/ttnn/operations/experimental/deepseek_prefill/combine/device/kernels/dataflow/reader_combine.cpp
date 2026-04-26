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
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

// Signal last element to writer to break out of loop
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-5)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_for_writer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(5);

    // Page counts (indices 6-9)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(6);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(7);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(8);
    constexpr uint32_t output_pages = get_compile_time_arg_val(9);

    // Page sizes (indices 10-13)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(13);

    // Operation parameters (indices 14-17)
    constexpr uint32_t num_chips = get_compile_time_arg_val(14);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(16);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(17);

    // Hidden dimension (index 18)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(18);

    // Aligned page sizes (indices 19-22)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(22);

    // Mesh information (indices 23-27)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(23);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(26);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(27);

    // Fabric configuration (indices 28-31)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(28);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(29);
    constexpr uint32_t num_links = get_compile_time_arg_val(30);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(31);

    // Batch configuration (index 32)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(32);
    // Number of dispatch groups (index 33)
    constexpr uint32_t num_dispatch_groups = get_compile_time_arg_val(33);

    // Expert region offsets tensor metadata (indices 34-37)
    constexpr uint32_t cb_expert_region_offsets_id = get_compile_time_arg_val(34);
    constexpr uint32_t expert_region_offsets_pages = get_compile_time_arg_val(35);
    constexpr uint32_t expert_region_offsets_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_expert_region_offsets_page_size = get_compile_time_arg_val(37);

    // Dispatch buffer total per-chip capacity (index 38) — used as overflow guard.
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(38);

    // TensorAccessorArgs for all 5 tensors (starting at index 39)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<39>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();
    constexpr auto expert_region_offsets_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

#if INIT_ZEROS
    // Zero-init args follow immediately after the TensorAccessorArgs block
    constexpr uint32_t zi_cb_id = get_compile_time_arg_val(expert_region_offsets_args.next_compile_time_args_offset());
    constexpr uint32_t num_total_idle_cores =
        get_compile_time_arg_val(expert_region_offsets_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t tile_layout_args_base = expert_region_offsets_args.next_compile_time_args_offset() + 2;
#else
    constexpr uint32_t tile_layout_args_base = expert_region_offsets_args.next_compile_time_args_offset();
#endif

#if IS_TILE_LAYOUT
    constexpr uint32_t num_idle_cores_group = get_compile_time_arg_val(tile_layout_args_base);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(tile_layout_args_base + 1);
    constexpr uint32_t cb_metadata_batch_id = get_compile_time_arg_val(tile_layout_args_base + 2);
    constexpr uint32_t cb_idle_c9_addr_scratch_id = get_compile_time_arg_val(tile_layout_args_base + 3);
#endif

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_region_offsets_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_semaphore_address = get_semaphore(zero_init_semaphore_id);
    uint32_t zero_init_barrier_address = get_semaphore(zero_init_barrier_semaphore_id);

    DPRINT_COMBINE << "Combine Reader: experts=[" << expert_start_idx << "," << expert_end_idx << ")"
                   << " linearized_mesh_coord=" << linearized_mesh_coord << ENDL();

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

#if INIT_ZEROS
    // Hybrid row zero-init: this core zeroes its assigned page range, then waits for idle row cores
    {
        uint32_t page_start = get_arg_val<uint32_t>(rt_args++);
        uint32_t page_end = get_arg_val<uint32_t>(rt_args++);
        uint32_t zi_done_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        uint32_t zi_done_sem_address = get_semaphore(zi_done_semaphore_id);

        fill_zero_buffer(zi_cb_id);
        uint32_t zero_buf = get_write_ptr(zi_cb_id);

        zero_pages(zero_buf, page_start, page_end, aligned_output_page_size, output_addr_gen);

        volatile tt_l1_ptr uint32_t* zi_done_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zi_done_sem_address);
        noc_semaphore_wait(zi_done_sem_ptr, num_total_idle_cores);
        noc_semaphore_set(zi_done_sem_ptr, 0);
    }
#endif

#if IS_TILE_LAYOUT
    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t idle_counter_l1_offset = get_write_ptr(cb_dispatched_metadata_id);
    uint32_t counter_ready_sem_l1_offset = get_semaphore(counter_ready_semaphore_id);
    uint64_t mcast_counter_noc_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, idle_counter_l1_offset);
    uint64_t mcast_counter_sem_noc_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, counter_ready_sem_l1_offset);

    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    volatile tt_l1_ptr uint32_t* data_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_semaphore_id));

    uint32_t start_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t start_sem_l1_offset = get_semaphore(start_semaphore_id);
    uint64_t idle_start_noc_addrs[num_idle_cores_group];
    uint32_t idle_noc_x[num_idle_cores_group];
    uint32_t idle_noc_y[num_idle_cores_group];
    for (uint32_t c = 0; c < num_idle_cores_group; c++) {
        idle_noc_x[c] = get_arg_val<uint32_t>(rt_args++);
        idle_noc_y[c] = get_arg_val<uint32_t>(rt_args++);
        idle_start_noc_addrs[c] = get_noc_addr(idle_noc_x[c], idle_noc_y[c], start_sem_l1_offset);
    }
#endif

    // Signal writer that zero-init is complete
    volatile tt_l1_ptr uint32_t* zero_init_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_semaphore_address);
    noc_semaphore_set(zero_init_sem_ptr, 1);

    // Wait for ALL writers (all cores) to complete init exchange.
    // Each writer signals all readers' barrier sems via noc_semaphore_inc,
    // so this reader waits for num_cores signals before proceeding.
    volatile tt_l1_ptr uint32_t* barrier_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_barrier_address);
    noc_semaphore_wait(barrier_sem_ptr, num_cores);
    noc_semaphore_set(barrier_sem_ptr, 0);

    // Read expert token counts
    const auto experts_tok_counter_addr_gen = TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr);
    cb_reserve_back(cb_experts_tok_counter_id, experts_tok_counter_pages);
    uint32_t counter_base_addr = get_write_ptr(cb_experts_tok_counter_id);
    for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
        noc_async_read_page(
            i, experts_tok_counter_addr_gen, counter_base_addr + i * aligned_experts_tok_counter_page_size);
    }
    noc_async_read_barrier();

    // Expert token counts: flat [num_routed_experts] array per device.
    // Decompose linearized_mesh_coord into (row, col) using physical mesh dims,
    // then map col -> dispatch_group_idx via modulo num_dispatch_groups.
    // This handles DP replicas (ndg < mesh_cols) where multiple columns share the same group.
    constexpr uint32_t mesh_row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t mesh_col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t dispatch_group_idx = mesh_col % num_dispatch_groups;
    constexpr uint32_t experts_per_dispatch_group = experts_per_chip * num_chips;
    constexpr uint32_t offset = dispatch_group_idx * experts_per_dispatch_group + mesh_row * experts_per_chip;
    // Multicast expert token counts + receive_buf_addr to all idle cores
#if IS_TILE_LAYOUT
    // Each sender multicasts token counts + its own receive_buf_addr + its c_10 scratch L1
    // address to its dedicated idle group. The mcast destination covers only this sender's
    // k_s idle cores (per-sender bounding box), so all senders can multicast in parallel.
    // Trailer layout (one l1_alignment region after counter_total_size bytes):
    //   [0]: receive_buf_addr  — sender's c_18 L1 offset (where idle NOC-writes untilized data)
    //   [1]: idle_c9_scratch_addr — sender's c_10 L1 offset (where idle NOC-writes its c_9 addr)
    {
        constexpr uint32_t counter_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;

        volatile tt_l1_ptr uint32_t* trailer_slot =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_base_addr + counter_total_size);
        trailer_slot[0] = get_write_ptr(cb_untilize_id);
        trailer_slot[1] = get_write_ptr(cb_idle_c9_addr_scratch_id);

        constexpr uint32_t mcast_total_size = counter_total_size + l1_alignment;
        uint32_t off = 0;
        while (off < mcast_total_size) {
            uint32_t chunk = ((mcast_total_size - off) > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE
                                                                                       : (mcast_total_size - off);
            noc_async_write_multicast(
                counter_base_addr + off, mcast_counter_noc_addr + off, chunk, num_idle_cores_group);
            off += chunk;
        }
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(mcast_counter_sem_noc_addr, 1, num_idle_cores_group);
        noc_async_atomic_barrier();
    }
#endif

    volatile tt_l1_ptr uint32_t* experts_tok_counter_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_base_addr) + offset;

    // Set up scratch buffers for batched reads
    cb_reserve_back(cb_dispatched_metadata_id, read_batch_size);
    uint32_t metadata_base = get_write_ptr(cb_dispatched_metadata_id);
    const auto dispatched_metadata_addr_gen =
        TensorAccessor(dispatched_metadata_args, dispatched_metadata_addr, aligned_dispatched_metadata_page_size);

#if IS_TILE_LAYOUT
    uint32_t untilize_base = get_write_ptr(cb_untilize_id);

    // ===== Idle c_9 address handshake =====
    // After the counter multicast above, each idle core knows where our c_10 scratch lives and
    // will write its get_write_ptr(c_9) L1 offset to slot core_id * sizeof(uint32_t).  They then
    // increment data_ready_sem (same sem reused for its normal per-batch role below).  Once we
    // see data_ready_sem == num_idle_cores_group, all idle c_9 addresses are in our c_10 scratch.
    noc_semaphore_wait(data_ready_sem_ptr, num_idle_cores_group);
    noc_semaphore_set(data_ready_sem_ptr, 0);

    // Copy the received L1 offsets out of c_10 into a plain uint32_t[] array, then build the
    // per-idle-core NOC addresses used by the batch loop below for the metadata unicast.
    uint32_t idle_c9_scratch_base = get_write_ptr(cb_idle_c9_addr_scratch_id);
    uint32_t idle_c9_addrs[num_idle_cores_group];
    uint64_t idle_metadata_noc_addrs[num_idle_cores_group];
    for (uint32_t c = 0; c < num_idle_cores_group; c++) {
        idle_c9_addrs[c] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idle_c9_scratch_base + c * sizeof(uint32_t));
        idle_metadata_noc_addrs[c] = get_noc_addr(idle_noc_x[c], idle_noc_y[c], idle_c9_addrs[c]);
    }
#else
    cb_reserve_back(cb_dispatched_buffer_id, read_batch_size);
    uint32_t buffer_base = get_write_ptr(cb_dispatched_buffer_id);
    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
#endif

    // Read expert region offsets directly from the host-provided tensor.
    // Layout matches expert_token_counts: this device's experts_per_chip slice lives at
    // [mesh_col, mesh_row, experts_per_chip] within a flat [num_routed_experts] page.
    const auto expert_region_offsets_addr_gen = TensorAccessor(expert_region_offsets_args, expert_region_offsets_addr);
    cb_reserve_back(cb_expert_region_offsets_id, expert_region_offsets_pages);
    uint32_t region_offsets_base_addr = get_write_ptr(cb_expert_region_offsets_id);
    for (uint32_t i = 0; i < expert_region_offsets_pages; i++) {
        noc_async_read_page(
            i, expert_region_offsets_addr_gen, region_offsets_base_addr + i * aligned_expert_region_offsets_page_size);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* expert_region_offsets_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region_offsets_base_addr) + offset;

    // Process each expert in assigned range
    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t start_page = expert_region_offsets_l1[local_expert];
        uint32_t expert_tokens = experts_tok_counter_l1[local_expert];
        // Clamp to the dispatch buffer capacity to mirror reader_dispatch's overflow guard:
        // dispatch silently drops tokens beyond max_dispatch_buffer_token_size, so reading
        // past it would pull stale/zero-init data and risk out-of-bounds DRAM access.
        if (start_page >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_page + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_page;
        }
        uint32_t end_page = start_page + expert_tokens;
        uint32_t num_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        DPRINT_COMBINE << "Expert=" << local_expert << " tokens=" << expert_tokens << ENDL();

#if IS_TILE_LAYOUT
#else
        // Prefetch first batch
        uint32_t first_batch_end = (start_page + read_batch_size < end_page) ? start_page + read_batch_size : end_page;
        uint32_t first_batch_count = first_batch_end - start_page;
        if (first_batch_count > 0) {
            for (uint32_t t = 0; t < first_batch_count; t++) {
                noc_async_read_page(
                    start_page + t, dispatched_buffer_addr_gen, buffer_base + t * aligned_dispatched_buffer_page_size);
                noc_async_read_page(
                    start_page + t,
                    dispatched_metadata_addr_gen,
                    metadata_base + t * aligned_dispatched_metadata_page_size);
            }
            noc_async_read_barrier();
        }
#endif

        for (uint32_t B = 0; B < num_batches; B++) {
            uint32_t batch_start = start_page + B * read_batch_size;
            uint32_t batch_end = (batch_start + read_batch_size < end_page) ? batch_start + read_batch_size : end_page;
            uint32_t batch_count = batch_end - batch_start;
            bool batch_did_local_write = false;

#if IS_TILE_LAYOUT
            uint32_t current_idle_core = B % num_idle_cores_group;

            // Read metadata FIRST (we need to inspect it to decide if idle can handle the whole
            // batch locally or whether we need untilized data back to do non-local routing).
            for (uint32_t t = 0; t < batch_count; t++) {
                noc_async_read_page(
                    batch_start + t,
                    dispatched_metadata_addr_gen,
                    metadata_base + t * aligned_dispatched_metadata_page_size);
            }
            noc_async_read_barrier();

            // Scan metadata for any non-local writes (dst_chip != our linearized_mesh_coord).
            bool has_non_local = false;
            for (uint32_t t = 0; t < batch_count; t++) {
                volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    metadata_base + t * aligned_dispatched_metadata_page_size);
                if (m[0] != linearized_mesh_coord) {
                    has_non_local = true;
                    break;
                }
            }

            if (has_non_local) {
                // Idle core only checks c_9[0] on the non-local path — skip the full metadata
                // unicast and just write the sentinel directly.
                noc_inline_dw_write(idle_metadata_noc_addrs[current_idle_core], ROUTE_INFO_SENTINEL);
            } else {
                // Unicast the full metadata batch to the idle core's c_9 so it can write each
                // row directly to output DRAM, then set c_9[0] = batch_count as the signal.
                uint32_t metadata_unicast_bytes = batch_count * aligned_dispatched_metadata_page_size;
                noc_async_write(metadata_base, idle_metadata_noc_addrs[current_idle_core], metadata_unicast_bytes);
                noc_async_write_barrier();
                noc_inline_dw_write(idle_metadata_noc_addrs[current_idle_core], batch_count);
            }

            // Signal the idle core that the metadata batch is ready in its c_9.
            noc_semaphore_inc(idle_start_noc_addrs[current_idle_core], 1);
            noc_async_atomic_barrier();

            // Always wait on data_ready — even if has_non_local is false — to keep start_sem
            // strictly 1:1 with idle's consume/reset.  Idle's `wait(>=1); set(0)` collapses
            // multiple pending increments into a single consume, so if we advance without
            // syncing, a later start_sem inc to the same idle core gets silently dropped and
            // the idle deadlocks waiting for a signal that never re-arrives.
            noc_semaphore_wait(data_ready_sem_ptr, 1);
            noc_semaphore_set(data_ready_sem_ptr, 0);
#endif

            // In TILE_LAYOUT the sender only routes when has_non_local is true — idle already
            // handled all-local batches directly. In ROW_MAJOR there is no idle offload.
#if IS_TILE_LAYOUT
            if (has_non_local)
#endif
            {
                for (uint32_t t = 0; t < batch_count; t++) {
#if IS_TILE_LAYOUT
                    uint32_t buffer_scratch_addr = untilize_base + t * aligned_output_page_size;
#else
                    uint32_t buffer_scratch_addr = buffer_base + t * aligned_dispatched_buffer_page_size;
#endif
                    uint32_t metadata_scratch_addr = metadata_base + t * aligned_dispatched_metadata_page_size;
                    volatile tt_l1_ptr uint32_t* metadata =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_scratch_addr);

                    auto dst_chip = metadata[0];
                    auto dst_token_idx = metadata[1];
                    auto dst_topk_indice = metadata[2];
                    uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;

                    if (dst_chip == linearized_mesh_coord) {
                        noc_async_write_page(output_page_idx, output_addr_gen, buffer_scratch_addr);
                        noc_async_writes_flushed();
                        batch_did_local_write = true;
                    } else {
                        if constexpr (is_1d_topology<topology>()) {
                            uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                            uint32_t distance =
                                manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);

                            // Push route info to writer
                            cb_reserve_back(cb_route_info_id, 1);
                            volatile tt_l1_ptr uint32_t* route_info =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
                            route_info[0] = route;
                            route_info[1] = distance;
                            route_info[2] = output_page_idx;
                            route_info[3] = 0;
                            cb_push_back(cb_route_info_id, 1);

                            // Push output payload to writer
                            cb_reserve_back(cb_output_for_writer_id, 1);
                            uint32_t output_dst = get_write_ptr(cb_output_for_writer_id);
#if IS_TILE_LAYOUT
                            noc_async_read(get_noc_addr(buffer_scratch_addr), output_dst, aligned_output_page_size);
#else
                            noc_async_read(
                                get_noc_addr(buffer_scratch_addr), output_dst, aligned_dispatched_buffer_page_size);
#endif
                            noc_async_read_barrier();
                            cb_push_back(cb_output_for_writer_id, 1);
                        }
                    }
                }

#if IS_TILE_LAYOUT
#else
                // Issue next batch reads BEFORE write barrier to overlap DMA reads with NOC writes
                uint32_t next_batch_start = batch_start + read_batch_size;
                bool has_next_batch = (next_batch_start < end_page);
                if (has_next_batch) {
                    uint32_t next_batch_end =
                        (next_batch_start + read_batch_size < end_page) ? next_batch_start + read_batch_size : end_page;
                    uint32_t next_batch_count = next_batch_end - next_batch_start;
                    for (uint32_t t = 0; t < next_batch_count; t++) {
                        noc_async_read_page(
                            next_batch_start + t,
                            dispatched_buffer_addr_gen,
                            buffer_base + t * aligned_dispatched_buffer_page_size);
                        noc_async_read_page(
                            next_batch_start + t,
                            dispatched_metadata_addr_gen,
                            metadata_base + t * aligned_dispatched_metadata_page_size);
                    }
                }
#endif

                if (batch_did_local_write) {
                    noc_async_write_barrier();
                }

#if IS_TILE_LAYOUT
#else
                if (has_next_batch) {
                    noc_async_read_barrier();
                }
#endif
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
