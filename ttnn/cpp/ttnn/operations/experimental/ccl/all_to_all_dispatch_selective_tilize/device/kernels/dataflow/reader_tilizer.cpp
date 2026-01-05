// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

// Helper to get multicast NOC address with proper coordinate ordering for NOC 0 vs NOC 1.
// NOC 0: start = (min_x, min_y), end = (max_x, max_y)
// NOC 1: start = (max_x, max_y), end = (min_x, min_y) - coordinates need to be swapped
FORCE_INLINE uint64_t get_safe_multicast_noc_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr,
    uint8_t noc = noc_index) {
    if (noc == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    } else {
        // For NOC 1, swap start and end coordinates
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr, noc);
    }
}

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void zero_buffer_async(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

void zero_buffer_barrier() { noc_async_read_barrier(); }

// DEBUG: Print E-D table contents
template <uint32_t ExpertsPerDevice, uint32_t DispatchDevices, uint32_t EntriesPerL1Alignment>
void print_ed_table(volatile tt_l1_ptr uint32_t* ed_table, uint32_t device_id, const char* label) {
    // Invalidate L1 cache to see updates from remote writes
    invalidate_l1_cache();

    DPRINT << "=== E-D Table: " << label << " (device " << device_id << ") ===" << ENDL();
    for (uint32_t local_expert = 0; local_expert < ExpertsPerDevice; local_expert++) {
        DPRINT << "Expert " << local_expert << ": [";
        for (uint32_t src_dev = 0; src_dev < DispatchDevices; src_dev++) {
            // Each entry is l1_alignment bytes apart, access first uint32 of each entry
            uint32_t entry_idx = (local_expert * DispatchDevices + src_dev) * EntriesPerL1Alignment;
            uint32_t val = ed_table[entry_idx];
            DPRINT << val;
            if (src_dev < DispatchDevices - 1) {
                DPRINT << ", ";
            }
        }
        DPRINT << "]" << ENDL();
    }
    DPRINT << "---" << ENDL();
}

template <uint32_t experts_per_device, uint32_t dispatch_devices, uint32_t entries_per_l1_alignment>
void poll_ed_table_loop(volatile tt_l1_ptr uint32_t* ed_table, uint32_t linearized_mesh_coord) {
    uint32_t poll_iteration = 0;
    while (true) {
        DPRINT << "Poll #" << poll_iteration << ENDL();
        print_ed_table<experts_per_device, dispatch_devices, entries_per_l1_alignment>(
            ed_table, linearized_mesh_coord, "polling");

        poll_iteration++;
        // Small delay between polls to avoid flooding DPRINT
        for (volatile uint32_t delay = 0; delay < 100000; delay++) {
        }
    }
}

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t e_d_buffer_id = get_named_compile_time_arg_val("e_d_buffer_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t metadata_page_size = get_named_compile_time_arg_val("metadata_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens_per_device = get_named_compile_time_arg_val("tokens_per_device");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");

    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_metadata_page_size = get_named_compile_time_arg_val("aligned_metadata_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");
    constexpr uint32_t max_indices_pages_per_packet = get_named_compile_time_arg_val("max_indices_pages_per_packet");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");

    // Multicast coordinates for signaling sender cores that E-D buffer is ready
    constexpr uint32_t sender_mcast_start_x = get_named_compile_time_arg_val("sender_mcast_start_x");
    constexpr uint32_t sender_mcast_start_y = get_named_compile_time_arg_val("sender_mcast_start_y");
    constexpr uint32_t sender_mcast_end_x = get_named_compile_time_arg_val("sender_mcast_end_x");
    constexpr uint32_t sender_mcast_end_y = get_named_compile_time_arg_val("sender_mcast_end_y");
    // Multicast coordinates for signaling tilizer cores
    constexpr uint32_t tilizer_mcast_start_x = get_named_compile_time_arg_val("tilizer_mcast_start_x");
    constexpr uint32_t tilizer_mcast_start_y = get_named_compile_time_arg_val("tilizer_mcast_start_y");
    constexpr uint32_t tilizer_mcast_end_x = get_named_compile_time_arg_val("tilizer_mcast_end_x");
    constexpr uint32_t tilizer_mcast_end_y = get_named_compile_time_arg_val("tilizer_mcast_end_y");
    constexpr uint32_t num_tilizer_cores = get_named_compile_time_arg_val("num_tilizer_cores");
    constexpr uint32_t num_sender_cores = get_named_compile_time_arg_val("num_sender_cores");
    constexpr uint32_t ed_buffer_ready_semaphore_id = get_named_compile_time_arg_val("ed_buffer_ready_semaphore_id");
    constexpr uint32_t ed_table_computed_semaphore_id =
        get_named_compile_time_arg_val("ed_table_computed_semaphore_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    uint32_t ed_addr = get_read_ptr(e_d_buffer_id);
    zero_buffer_async(ed_addr, 2 * experts_per_device * dispatch_devices * l1_alignment * sizeof(uint32_t));
    size_t rt_args_idx = 0;
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);           // 0
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 1
    uint32_t output_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    uint32_t indices_sent_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);  // 3
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);          // 4
    bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);         // 5
    zero_buffer_barrier();

    // DEBUG: Print E-D table right after zeroing to verify it's actually zeroed
    constexpr uint32_t entries_per_l1_alignment = l1_alignment / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* ed_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ed_addr);
    // print_ed_table<experts_per_device, dispatch_devices, entries_per_l1_alignment>(
    //     ed_table, linearized_mesh_coord, "after zero");

    // Signal all sender cores that E-D buffer has been zeroed
    // Set local semaphore to 1, then multicast write to all sender cores
    if (is_drain_tilizer_core) {
        uint32_t ed_ready_sem_addr = get_semaphore(ed_buffer_ready_semaphore_id);
        volatile tt_l1_ptr uint32_t* ed_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ed_ready_sem_addr);
        noc_semaphore_set(ed_ready_sem_ptr, 1);

        // Use safe multicast address that handles NOC 0 vs NOC 1 coordinate ordering
        uint64_t sender_mcast_noc_addr = get_safe_multicast_noc_addr(
            sender_mcast_start_x, sender_mcast_start_y, sender_mcast_end_x, sender_mcast_end_y, ed_ready_sem_addr);

        // Multicast write the semaphore value (1) to all sender cores
        // Sender cores wait for value 1 to know E-D buffer is ready
        noc_semaphore_set_multicast(ed_ready_sem_addr, sender_mcast_noc_addr, num_sender_cores);
        noc_async_write_barrier();
    }

    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto output_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_tensor_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    // scores and indices tensors use same page size
    const auto output_scores_tensor_addr_gen =
        TensorAccessor(output_scores_args, output_scores_tensor_address, indices_page_size);

    // Wait for all fabric writer cores to signal that indices/scores have been sent
    // This ensures the metadata all-gather is complete before post-processing
    // Only drain_tilizer_cores receive the atomic increments and do the post-processing
    if (is_drain_tilizer_core) {
        noc_semaphore_wait((uint32_t*)indices_sent_semaphore_address, dispatch_devices - 1);
        noc_semaphore_set((uint32_t*)indices_sent_semaphore_address, 0);

        // POST-PROCESSING STAGE:
        // Read in metadata into indices cb and update the ground truth E-D table offset into the second half of the
        // E-D buffer. Must be done AFTER semaphore wait to ensure all-gather is complete.
        uint32_t base_ed_table_offset = experts_per_device * dispatch_devices * l1_alignment;
        for (uint32_t device_id = 0; device_id < dispatch_devices; device_id++) {
            for (uint32_t indices_page = 0; indices_page < indices_pages; indices_page++) {
                uint32_t metadata_page = device_id * indices_pages + indices_page;
                cb_reserve_back(indices_tensor_cb_id, 1);
                noc_async_read_page(metadata_page, metadata_tensor_addr_gen, get_write_ptr(indices_tensor_cb_id));
                noc_async_read_barrier();
                cb_push_back(indices_tensor_cb_id, 1);
            }
        }
    }

    // DEBUG: Polling loop to verify E-D buffer semaphore increments are landing
    // poll_ed_table_loop<experts_per_device, dispatch_devices, entries_per_l1_alignment>(ed_table,
    // linearized_mesh_coord);

    // Tilizer cores wait for the E-D table computation to be complete
    uint32_t ed_computed_sem_addr = get_semaphore(ed_table_computed_semaphore_id);
    noc_semaphore_wait((uint32_t*)ed_computed_sem_addr, 1);

    // All tilizer cores poll their ed_table against the ground truth (final_ed_table)
    // - Drain tilizer core's ed_table is updated by remote devices (via all-to-all dispatch)
    // - Non-drain tilizer cores' ed_tables are updated by the drain core (via multicast)
    // When drain core sees a match, it multicasts a semaphore increment to non-drain cores
    uint32_t base_ed_table_offset = ed_addr + experts_per_device * dispatch_devices * l1_alignment;
    volatile tt_l1_ptr uint32_t* final_ed_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_ed_table_offset);

    // Track which entries have already matched (to avoid re-printing/re-signaling)
    // Use a bitmask approach - each bit represents one (expert, device) pair
    // For simplicity, we'll use a simple array of booleans
    constexpr uint32_t total_entries = experts_per_device * dispatch_devices;
    bool entry_matched[total_entries];
    for (uint32_t i = 0; i < total_entries; i++) {
        entry_matched[i] = false;
    }

    uint32_t entries_matched = 0;

    // Poll until all entries match
    while (entries_matched < total_entries) {
        // Invalidate L1 cache to see updates from remote writes
        invalidate_l1_cache();

        for (uint32_t local_expert = 0; local_expert < experts_per_device; local_expert++) {
            for (uint32_t src_dev = 0; src_dev < dispatch_devices; src_dev++) {
                uint32_t flat_idx = local_expert * dispatch_devices + src_dev;

                // Skip if already matched
                if (entry_matched[flat_idx]) {
                    continue;
                }

                // Calculate L1 address of this ed_table entry
                uint32_t ed_entry_l1_addr = ed_addr + flat_idx * l1_alignment;
                uint32_t current_val = ed_table[flat_idx * entries_per_l1_alignment];
                uint32_t expected_val = final_ed_table[flat_idx * entries_per_l1_alignment];

                if (current_val == expected_val) {
                    // Entry matched!
                    entry_matched[flat_idx] = true;
                    entries_matched++;

                    // Drain core multicasts the ed_table entry value to non-drain cores
                    // This directly updates the ed_table on non-drain cores since they
                    // are not being incremented by remote devices
                    if (is_drain_tilizer_core) {
                        if constexpr (num_tilizer_cores > 1) {
                            uint64_t ed_entry_mcast_addr = get_safe_multicast_noc_addr(
                                tilizer_mcast_start_x,
                                tilizer_mcast_start_y,
                                tilizer_mcast_end_x,
                                tilizer_mcast_end_y,
                                ed_entry_l1_addr);
                            // Write 4 bytes (the ed_table entry value) to all tilizer cores
                            noc_semaphore_set_multicast_loopback_src(
                                ed_entry_l1_addr, ed_entry_mcast_addr, num_tilizer_cores);
                        }
                    }

                    // Print the matched value
                    DPRINT << "E[" << local_expert << "][D" << src_dev << "] = " << current_val << ENDL();
                }
            }
        }
    }

    // Final barrier to ensure all multicasts complete
    if (is_drain_tilizer_core) {
        if constexpr (num_tilizer_cores > 1) {
            noc_async_write_barrier();
        }
    }

    DPRINT << "=== All E-D entries matched! ===" << ENDL();
}
