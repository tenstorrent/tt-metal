// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch reader kernel (combined metadata+payload variant)
// Handles DRAM reads, metadata writing, local DRAM writes, and route/distance
// computation for remote experts. Pushes combined pages and route info to
// the writer via CBs for fabric sends.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_combined_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-10)
    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(7);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(8);
    constexpr uint32_t combined_output_pages = get_compile_time_arg_val(9);
    constexpr uint32_t counter_pages = get_compile_time_arg_val(10);

    // Page sizes (indices 11-16)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t combined_output_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t counter_page_size = get_compile_time_arg_val(16);

    // Operation parameters (indices 17-25)
    constexpr uint32_t num_devices = get_compile_time_arg_val(17);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(18);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(19);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(20);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(22);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(23);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(24);
    constexpr uint32_t padded_metadata_bytes = get_compile_time_arg_val(25);

    // Mesh information (indices 26-30)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(26);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(27);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(28);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(29);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(30);

    // Aligned page sizes (indices 31-36)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(31);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_combined_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_counter_page_size = get_compile_time_arg_val(36);

    // Fabric configuration (indices 37-40)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(38);
    constexpr uint32_t num_links = get_compile_time_arg_val(39);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(40);

    // Additional CB IDs (indices 41-42)
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(41);
    constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(42);

    constexpr uint32_t combined_cb_page_size = padded_metadata_bytes + aligned_input_page_size;

    // TensorAccessorArgs for 6 tensors (starting at index 43)
    constexpr auto input_args = TensorAccessorArgs<43>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto combined_output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto counter_args = TensorAccessorArgs<combined_output_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);            // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);          // 1
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args++);          // 2
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args++);          // 3
    uint32_t combined_output_address = get_arg_val<uint32_t>(rt_args++);         // 4
    uint32_t counter_tensor_address = get_arg_val<uint32_t>(rt_args++);          // 5
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args++);  // 6
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args++);          // 7
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args++);                 // 8
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args++);                   // 9
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args++);               // 10
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args++);              // 11
    uint32_t assigned_direction = get_arg_val<uint32_t>(rt_args++);              // 12
    uint32_t assigned_link = get_arg_val<uint32_t>(rt_args++);                   // 13
    uint32_t batch_ready_sem_id = get_arg_val<uint32_t>(rt_args++);              // 14
    uint32_t batch_consumed_sem_id = get_arg_val<uint32_t>(rt_args++);           // 15
    uint32_t num_secondary_cores = get_arg_val<uint32_t>(rt_args++);             // 16
    uint32_t primary_noc_x = get_arg_val<uint32_t>(rt_args++);                   // 17
    uint32_t primary_noc_y = get_arg_val<uint32_t>(rt_args++);                   // 18
    uint32_t mcast_noc_x_start = get_arg_val<uint32_t>(rt_args++);               // 19
    uint32_t mcast_noc_y_start = get_arg_val<uint32_t>(rt_args++);               // 20
    uint32_t mcast_noc_x_end = get_arg_val<uint32_t>(rt_args++);                 // 21
    uint32_t mcast_noc_y_end = get_arg_val<uint32_t>(rt_args++);                 // 22

    constexpr uint32_t DIR_INVALID = 0xFFFFFFFF;
    bool is_primary = (dispatch_core_idx == 0);
    bool has_secondaries = (num_secondary_cores > 0);
    bool is_local_only = (assigned_direction == DIR_INVALID);
    uint32_t effective_num_links = (num_dispatch_cores > 1) ? (num_dispatch_cores / 2) : 1;

    uint32_t sem_ready_l1_addr = get_semaphore(batch_ready_sem_id);
    uint32_t sem_consumed_l1_addr = get_semaphore(batch_consumed_sem_id);
    volatile tt_l1_ptr uint32_t* sem_ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_l1_addr);
    volatile tt_l1_ptr uint32_t* sem_consumed_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_consumed_l1_addr);
    uint64_t primary_sem_consumed_noc_addr = get_noc_addr(primary_noc_x, primary_noc_y, sem_consumed_l1_addr);
    uint32_t ready_count = 0;
    uint32_t consumed_count = 0;

    DPRINT_DISPATCH << "Reader combined kernel: tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " dispatch_core=" << dispatch_core_idx << "/" << num_dispatch_cores
                    << " is_primary=" << (uint32_t)is_primary << " assigned_dir=" << assigned_direction
                    << " assigned_link=" << assigned_link << " num_secondary=" << num_secondary_cores << ENDL();

    // Read offsets tensor into local scratch (reader-only, no push to writer)
    int32_t* offsets;
    {
        DeviceZoneScopedN("dispatch-combined-read-offsets");
        const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address, offsets_page_size);
        cb_reserve_back(cb_offsets_id, offsets_pages);
        uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
        for (uint32_t i = 0; i < offsets_pages; i++) {
            noc_async_read_page(i, offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
        }
        noc_async_read_barrier();
        offsets = (int32_t*)offsets_base_addr;
    }

    uint32_t debug_local_count = 0;
    uint32_t debug_1hop_count = 0;
    uint32_t debug_2hop_count = 0;

    constexpr uint32_t read_batch_size = 8;

    // Batched scratch buffers for grouped DRAM reads (matches non-combined kernel's batch approach)
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);

    cb_reserve_back(cb_weights_id, read_batch_size);
    uint32_t weights_base = get_write_ptr(cb_weights_id);

    cb_reserve_back(cb_scratch_id, read_batch_size);
    uint32_t scratch_base = get_write_ptr(cb_scratch_id);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    const auto combined_addr_gen =
        TensorAccessor(combined_output_args, combined_output_address, aligned_combined_output_page_size);

    uint32_t fabric_send_counter = 0;

    // Prefetch first batch
    uint32_t first_batch_end =
        (token_start_idx + read_batch_size < token_end_idx) ? token_start_idx + read_batch_size : token_end_idx;
    uint32_t first_batch_count = first_batch_end - token_start_idx;
    if (first_batch_count > 0) {
        if (is_primary) {
            DeviceZoneScopedN("dispatch-combined-read-dram");
            for (uint32_t t = 0; t < first_batch_count; t++) {
                noc_async_read_page(
                    token_start_idx + t,
                    input_addr_gen,
                    scratch_base + t * combined_cb_page_size + padded_metadata_bytes);
                noc_async_read_page(
                    token_start_idx + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                noc_async_read_page(
                    token_start_idx + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }
            noc_async_read_barrier();
            if (has_secondaries) {
                uint64_t mcast_scratch = get_noc_multicast_addr(
                    mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, scratch_base);
                noc_async_write_multicast(
                    scratch_base, mcast_scratch, first_batch_count * combined_cb_page_size, num_secondary_cores);
                uint64_t mcast_indices = get_noc_multicast_addr(
                    mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, indices_base);
                noc_async_write_multicast(
                    indices_base, mcast_indices, first_batch_count * aligned_indices_page_size, num_secondary_cores);
                uint64_t mcast_weights = get_noc_multicast_addr(
                    mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, weights_base);
                noc_async_write_multicast(
                    weights_base, mcast_weights, first_batch_count * aligned_weights_page_size, num_secondary_cores);
                noc_async_write_barrier();
                ready_count++;
                uint64_t mcast_sem_ready = get_noc_multicast_addr(
                    mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, sem_ready_l1_addr);
                noc_semaphore_inc_multicast(mcast_sem_ready, 1, num_secondary_cores);
            }
        } else {
            DeviceZoneScopedN("dispatch-combined-wait-mcast");
            ready_count++;
            noc_semaphore_wait(sem_ready_ptr, ready_count);
            noc_semaphore_inc(primary_sem_consumed_noc_addr, 1);
        }
    }

    uint32_t local_send_counter = 0;

    {
        DeviceZoneScopedN("dispatch-combined-reader-token-loop");
        for (uint32_t batch_start = token_start_idx; batch_start < token_end_idx; batch_start += read_batch_size) {
            uint32_t batch_end =
                (batch_start + read_batch_size < token_end_idx) ? batch_start + read_batch_size : token_end_idx;
            uint32_t batch_count = batch_end - batch_start;

            bool batch_did_local_write = false;

            for (uint32_t t = 0; t < batch_count; t++) {
                uint32_t token_idx = batch_start + t;
                uint32_t scratch_addr = scratch_base + t * combined_cb_page_size;

                DPRINT_DISPATCH << "Reader processing token_idx: " << token_idx << ENDL();

                int32_t* indices = (int32_t*)(indices_base + t * aligned_indices_page_size);
                uint16_t* weights = (uint16_t*)(weights_base + t * aligned_weights_page_size);

                for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                    auto routed_expert = indices[k];
                    auto expert_chip = routed_expert / experts_per_chip;
                    auto expert_index_within_chip = routed_expert % experts_per_chip;
                    auto& offset = offsets[routed_expert];
                    auto page_idx = expert_index_within_chip * max_dispatched_tokens_per_expert + offset;

                    if (expert_chip == linearized_mesh_coord) {
                        bool my_turn = (local_send_counter % num_dispatch_cores == dispatch_core_idx);
                        local_send_counter++;
                        if (my_turn) {
                            debug_local_count++;
                            DPRINT_DISPATCH << "    Local dispatch expert=" << routed_expert << ENDL();
                            {
                                DeviceZoneScopedN("dispatch-combined-write-metadata");
                                volatile tt_l1_ptr int32_t* metadata =
                                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(scratch_addr);
                                metadata[0] = linearized_mesh_coord;
                                metadata[1] = token_idx;
                                metadata[2] = k;
                                metadata[3] = routed_expert;
                                metadata[4] = weights[k];
                            }
                            {
                                DeviceZoneScopedN("dispatch-combined-local");
                                noc_async_write_page(page_idx, combined_addr_gen, scratch_addr);
                                noc_async_writes_flushed();
                                batch_did_local_write = true;
                            }
                        }
                    } else {
                        if constexpr (is_1d_topology<topology>()) {
                            uint32_t route =
                                get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                            uint32_t distance =
                                manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                            uint32_t link = fabric_send_counter % effective_num_links;
                            fabric_send_counter++;

                            bool my_turn = (route == assigned_direction && link == assigned_link);

                            if (my_turn) {
                                if (distance == 1) {
                                    debug_1hop_count++;
                                } else {
                                    debug_2hop_count++;
                                }
                                DPRINT_DISPATCH << "    Remote dispatch expert=" << routed_expert << " route=" << route
                                                << " dist=" << distance << ENDL();
                                {
                                    DeviceZoneScopedN("dispatch-combined-write-metadata");
                                    volatile tt_l1_ptr int32_t* metadata =
                                        reinterpret_cast<volatile tt_l1_ptr int32_t*>(scratch_addr);
                                    metadata[0] = linearized_mesh_coord;
                                    metadata[1] = token_idx;
                                    metadata[2] = k;
                                    metadata[3] = routed_expert;
                                    metadata[4] = weights[k];
                                }
                                {
                                    DeviceZoneScopedN("dispatch-combined-push-route-info");
                                    cb_reserve_back(cb_route_info_id, 1);
                                    uint32_t route_info_addr = get_write_ptr(cb_route_info_id);
                                    volatile tt_l1_ptr uint32_t* route_info =
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_info_addr);
                                    route_info[0] = route;
                                    route_info[1] = distance;
                                    route_info[2] = page_idx;
                                    route_info[3] = 0;
                                    cb_push_back(cb_route_info_id, 1);
                                }
                                {
                                    DeviceZoneScopedN("dispatch-combined-push-combined");
                                    cb_reserve_back(cb_combined_id, 1);
                                    uint32_t combined_write_addr = get_write_ptr(cb_combined_id);
                                    noc_async_read(
                                        get_noc_addr(scratch_addr), combined_write_addr, combined_cb_page_size);
                                    noc_async_read_barrier();
                                    cb_push_back(cb_combined_id, 1);
                                }
                            }
                        }
                    }

                    offset++;
                }
            }

            // Issue next batch reads before write barrier so read channel
            // overlaps with write channel (local dispatch completion)
            uint32_t next_batch_start = batch_start + read_batch_size;
            bool has_next_batch = (next_batch_start < token_end_idx);
            uint32_t next_batch_count = 0;
            if (has_next_batch) {
                uint32_t next_batch_end = (next_batch_start + read_batch_size < token_end_idx)
                                              ? next_batch_start + read_batch_size
                                              : token_end_idx;
                next_batch_count = next_batch_end - next_batch_start;

                if (is_primary) {
                    if (has_secondaries) {
                        consumed_count += num_secondary_cores;
                        noc_semaphore_wait(sem_consumed_ptr, consumed_count);
                    }
                    DeviceZoneScopedN("dispatch-combined-read-dram");
                    for (uint32_t t = 0; t < next_batch_count; t++) {
                        noc_async_read_page(
                            next_batch_start + t,
                            input_addr_gen,
                            scratch_base + t * combined_cb_page_size + padded_metadata_bytes);
                        noc_async_read_page(
                            next_batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                        noc_async_read_page(
                            next_batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
                    }
                } else {
                    DeviceZoneScopedN("dispatch-combined-wait-mcast");
                    ready_count++;
                    noc_semaphore_wait(sem_ready_ptr, ready_count);
                }
            }

            if (batch_did_local_write) {
                DeviceZoneScopedN("dispatch-combined-write-barrier");
                noc_async_write_barrier();
            }

            if (has_next_batch) {
                if (is_primary) {
                    noc_async_read_barrier();
                    if (has_secondaries) {
                        uint64_t mcast_scratch = get_noc_multicast_addr(
                            mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, scratch_base);
                        noc_async_write_multicast(
                            scratch_base, mcast_scratch, next_batch_count * combined_cb_page_size, num_secondary_cores);
                        uint64_t mcast_indices = get_noc_multicast_addr(
                            mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, indices_base);
                        noc_async_write_multicast(
                            indices_base,
                            mcast_indices,
                            next_batch_count * aligned_indices_page_size,
                            num_secondary_cores);
                        uint64_t mcast_weights = get_noc_multicast_addr(
                            mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, weights_base);
                        noc_async_write_multicast(
                            weights_base,
                            mcast_weights,
                            next_batch_count * aligned_weights_page_size,
                            num_secondary_cores);
                        noc_async_write_barrier();
                        ready_count++;
                        uint64_t mcast_sem_ready = get_noc_multicast_addr(
                            mcast_noc_x_start, mcast_noc_y_start, mcast_noc_x_end, mcast_noc_y_end, sem_ready_l1_addr);
                        noc_semaphore_inc_multicast(mcast_sem_ready, 1, num_secondary_cores);
                    }
                } else {
                    noc_semaphore_inc(primary_sem_consumed_noc_addr, 1);
                }
            }
        }
    }

    DPRINT << "DISPATCH STATS: local=" << debug_local_count << " 1hop=" << debug_1hop_count
           << " 2hop=" << debug_2hop_count << ENDL();

    // Push sentinel to signal writer that all dispatches are done
    {
        DeviceZoneScopedN("dispatch-combined-push-sentinel");
        cb_reserve_back(cb_route_info_id, 1);
        uint32_t route_info_addr = get_write_ptr(cb_route_info_id);
        volatile tt_l1_ptr uint32_t* route_info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_info_addr);
        route_info[0] = ROUTE_INFO_SENTINEL;
        route_info[1] = 0;
        route_info[2] = 0;
        route_info[3] = 0;
        cb_push_back(cb_route_info_id, 1);
    }
}
