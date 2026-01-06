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

// Tile indexing helpers for 32x32 tiles with 4 faces (16x16 each)
// Face layout in memory: top-left, top-right, bottom-left, bottom-right
template <typename DataType>
FORCE_INLINE DataType* tile_row_offset(DataType* indices_address, uint32_t row) {
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t num_face_width = 2;
    uint32_t offset = 0;
    uint32_t local_row = row;
    if (row >= FaceHeight) {
        offset += num_face_width * FaceHeight * FaceWidth;  // Skip top two faces
        local_row -= FaceHeight;
    }
    offset += local_row * FaceWidth;
    return (DataType*)(indices_address + offset);
}

template <typename DataType>
FORCE_INLINE DataType* tile_col_offset(DataType* indices_address, uint32_t col) {
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    uint32_t offset = 0;
    uint32_t local_col = col;
    if (col >= FaceWidth) {
        offset += FaceHeight * FaceWidth;  // Skip to right face
        local_col -= FaceWidth;
    }
    offset += local_col;
    return (DataType*)(indices_address + offset);
}

// Process a chunk of tokens that have arrived for a given (local_expert, src_dev) pair.
// Scans metadata to find which tokens selected this expert, reads them, and pushes to CB.
// All token reads are issued first, then a single barrier is done before pushing all pages.
// Always reserves/pushes TokensPerChunk for simpler synchronization with compute.
template <
    uint32_t TileHeight,
    uint32_t SelectedExpertsK,
    uint32_t ExpertsPerDevice,
    uint32_t TokensPerDevice,
    uint32_t TilizerInputCbId,
    uint32_t AlignedOutputPageSize,
    uint32_t TokensPerChunk,
    typename MetadataAddrGenT,
    typename OutputAddrGenT>
FORCE_INLINE void process_chunk(
    uint32_t local_expert,
    uint32_t src_dev,
    uint32_t chunk_boundary,
    uint32_t linearized_mesh_coord,
    uint32_t indices_pages,
    uint32_t& last_scanned_token,
    uint32_t& tokens_found_count,
    uint32_t metadata_cb_addr,
    const MetadataAddrGenT& metadata_tensor_addr_gen,
    const OutputAddrGenT& output_tensor_addr_gen,
    uint16_t* devices_for_experts,
    uint32_t tilizer_subtoken_offset,
    uint32_t tilizer_subtoken_size) {
    // Calculate how many tokens we need to find in this chunk
    uint32_t tokens_to_find = chunk_boundary - tokens_found_count;
    uint32_t tokens_found_in_chunk = 0;

    // Metadata pages for src_dev are at: src_dev * indices_pages to (src_dev + 1) * indices_pages - 1
    uint32_t base_metadata_page = src_dev * indices_pages;

    // Start scanning from where we left off
    uint32_t scan_token = last_scanned_token;

    DPRINT << "Chunk E[" << local_expert << "][D" << src_dev << "]: finding " << tokens_to_find
           << " tokens starting from token " << scan_token << ENDL();

    // Always reserve TokensPerChunk for simpler synchronization with compute kernel
    // Extra slots in last iteration will contain garbage but won't affect results
    cb_reserve_back(TilizerInputCbId, TokensPerChunk);
    uint32_t tilizer_cb_base_addr = get_write_ptr(TilizerInputCbId);

    while (tokens_found_in_chunk < tokens_to_find && scan_token < TokensPerDevice) {
        uint32_t page_in_src = scan_token / TileHeight;
        uint32_t row_in_page = scan_token % TileHeight;

        // Read this metadata page
        uint32_t metadata_page = base_metadata_page + page_in_src;
        noc_async_read_page(metadata_page, metadata_tensor_addr_gen, metadata_cb_addr);
        noc_async_read_barrier();

        uint16_t* page_ptr = reinterpret_cast<uint16_t*>(metadata_cb_addr);

        // Process rows in this page starting from row_in_page
        while (row_in_page < TileHeight && tokens_found_in_chunk < tokens_to_find) {
            uint32_t token_idx = page_in_src * TileHeight + row_in_page;
            if (token_idx >= TokensPerDevice) {
                break;
            }

            // Get pointer to this row's expert indices
            uint16_t* row_ptr = tile_row_offset(page_ptr, row_in_page);

            // Check each of the K selected experts for this token
            for (uint32_t k = 0; k < SelectedExpertsK; k++) {
                uint16_t expert_id = tile_col_offset(row_ptr, k)[0];
                uint16_t device_for_expert = devices_for_experts[expert_id];

                // Check if this expert is on our device
                if (device_for_expert == linearized_mesh_coord) {
                    uint32_t expert_local_idx = expert_id % ExpertsPerDevice;

                    // Check if this matches the local_expert we're processing
                    if (expert_local_idx == local_expert) {
                        // Found a matching token!
                        // Output offset = token_idx + src_dev * tokens_per_device
                        uint32_t output_offset = token_idx + src_dev * TokensPerDevice;

                        DPRINT << "  Token " << token_idx << " from D" << src_dev << " selected expert " << expert_id
                               << " (local " << local_expert << ") -> output[" << output_offset << "]" << ENDL();

                        // Calculate write address for this token in the CB
                        // Each token slot is tilizer_subtoken_size bytes
                        uint32_t tilizer_cb_write_addr =
                            tilizer_cb_base_addr + tokens_found_in_chunk * tilizer_subtoken_size;

                        // Issue async read for this token's subtoken portion
                        // Token is at page output_offset, we read from tilizer_subtoken_offset within the page
                        uint64_t token_noc_addr =
                            get_noc_addr(output_offset, output_tensor_addr_gen, tilizer_subtoken_offset);
                        noc_async_read(token_noc_addr, tilizer_cb_write_addr, tilizer_subtoken_size);

                        tokens_found_in_chunk++;
                        break;  // Found match for this token, move to next token
                    }
                }
            }

            row_in_page++;
            scan_token++;
        }
    }

    // Wait for all token reads to complete, then push all pages at once
    // Always push TokensPerChunk for simpler synchronization with compute kernel
    noc_async_read_barrier();
    cb_push_back(TilizerInputCbId, TokensPerChunk);

    // Update tracking variables
    last_scanned_token = scan_token;
    tokens_found_count = chunk_boundary;
}

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
    DPRINT << "Reader tilizer core started" << ENDL();
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t e_d_buffer_id = get_named_compile_time_arg_val("e_d_buffer_id");
    constexpr uint32_t tilizer_input_cb_id = get_named_compile_time_arg_val("tilizer_input_cb_id");

    constexpr uint32_t tilizer_subtoken_bytes_aligned =
        get_named_compile_time_arg_val("tilizer_subtoken_bytes_aligned");

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
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");
    constexpr uint32_t max_indices_pages_per_packet = get_named_compile_time_arg_val("max_indices_pages_per_packet");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");
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
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;

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
    uint32_t tilizer_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);         // 6
    uint32_t tilizer_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);           // 7
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
    // Signaling happens at chunk boundaries (tokens_per_chunk) for pipelining
    constexpr uint32_t ed_table_page_size = experts_per_device * dispatch_devices * l1_alignment;
    uint32_t ground_truth_offset = ed_addr + ed_table_page_size;
    uint32_t temp_buffer_offset = ed_addr + 2 * ed_table_page_size;  // Page 2: temp buffer for mcast
    volatile tt_l1_ptr uint32_t* final_ed_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ground_truth_offset);
    volatile tt_l1_ptr uint32_t* temp_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(temp_buffer_offset);

    // Track last signaled value for each entry (all cores need this to detect new chunks)
    constexpr uint32_t total_entries = experts_per_device * dispatch_devices;
    uint32_t last_signaled[total_entries];
    for (uint32_t i = 0; i < total_entries; i++) {
        last_signaled[i] = 0;
    }

    uint32_t entries_matched = 0;
    // Track the last scanned token index for each (local_expert, src_dev) pair
    // This allows us to resume scanning from where we left off and avoid re-reading pages
    uint32_t last_scanned_token[experts_per_device][dispatch_devices];
    for (uint32_t e = 0; e < experts_per_device; e++) {
        for (uint32_t d = 0; d < dispatch_devices; d++) {
            last_scanned_token[e][d] = 0;
        }
    }
    // Track how many matching tokens we've found for each (local_expert, src_dev) pair
    uint32_t tokens_found_count[experts_per_device][dispatch_devices];
    for (uint32_t e = 0; e < experts_per_device; e++) {
        for (uint32_t d = 0; d < dispatch_devices; d++) {
            tokens_found_count[e][d] = 0;
        }
    }

    // Reserve CB space once for metadata reading - we'll reuse this buffer
    cb_reserve_back(indices_tensor_cb_id, 1);
    uint32_t metadata_cb_addr = get_write_ptr(indices_tensor_cb_id);

    // Wait for writer_tilizer (BRISC) to read the mapping tensor into the CB
    // The writer produces pages, we consume by waiting for them all to arrive
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);
    uint16_t* devices_for_experts = reinterpret_cast<uint16_t*>(get_read_ptr(mapping_tensor_cb_id));

    // Poll until all entries reach their ground truth values
    while (entries_matched < total_entries) {
        // Invalidate L1 cache to see updates from remote writes
        invalidate_l1_cache();

        for (uint32_t local_expert = 0; local_expert < experts_per_device; local_expert++) {
            for (uint32_t src_dev = 0; src_dev < dispatch_devices; src_dev++) {
                uint32_t flat_idx = local_expert * dispatch_devices + src_dev;
                uint32_t ground_truth_val = final_ed_table[flat_idx * entries_per_l1_alignment];

                // Skip if this entry has already reached ground truth
                if (last_signaled[flat_idx] >= ground_truth_val) {
                    continue;
                }

                uint32_t current_val = ed_table[flat_idx * entries_per_l1_alignment];

                if (is_drain_tilizer_core) {
                    // Drain core: signal at each chunk boundary
                    // Calculate next chunk target: next multiple of tokens_per_chunk, capped at ground_truth
                    uint32_t next_chunk_boundary =
                        ((last_signaled[flat_idx] / tokens_per_chunk) + 1) * tokens_per_chunk;
                    if (next_chunk_boundary > ground_truth_val) {
                        next_chunk_boundary = ground_truth_val;
                    }

                    // Process all chunk boundaries that current_val has passed
                    while (current_val >= next_chunk_boundary && last_signaled[flat_idx] < ground_truth_val) {
                        // Calculate L1 address of this ed_table entry for mcast destination
                        uint32_t ed_entry_l1_addr = ed_addr + flat_idx * l1_alignment;

                        // Write the chunk boundary value to temp buffer, then mcast to all tilizer cores
                        if constexpr (num_tilizer_cores > 1) {
                            temp_buffer[0] = next_chunk_boundary;
                            uint64_t ed_entry_mcast_addr = get_safe_multicast_noc_addr(
                                tilizer_mcast_start_x,
                                tilizer_mcast_start_y,
                                tilizer_mcast_end_x,
                                tilizer_mcast_end_y,
                                ed_entry_l1_addr);
                            // Mcast the temp buffer value to all tilizer cores' ed_table entry
                            noc_semaphore_set_multicast_loopback_src(
                                temp_buffer_offset, ed_entry_mcast_addr, num_tilizer_cores);
                        }

                        // CHUNK PROCESSING: Scan metadata to find tokens that selected this local_expert
                        process_chunk<
                            tile_height,
                            selected_experts_k,
                            experts_per_device,
                            tokens_per_device,
                            tilizer_input_cb_id,
                            aligned_output_page_size,
                            tokens_per_chunk>(
                            local_expert,
                            src_dev,
                            next_chunk_boundary,
                            linearized_mesh_coord,
                            indices_pages,
                            last_scanned_token[local_expert][src_dev],
                            tokens_found_count[local_expert][src_dev],
                            metadata_cb_addr,
                            metadata_tensor_addr_gen,
                            output_tensor_addr_gen,
                            devices_for_experts,
                            tilizer_subtoken_offset,
                            tilizer_subtoken_size);

                        // Update last_signaled and print
                        last_signaled[flat_idx] = next_chunk_boundary;
                        DPRINT << "E[" << local_expert << "][D" << src_dev << "] = " << next_chunk_boundary << ENDL();

                        // Check if we've reached ground truth
                        if (last_signaled[flat_idx] >= ground_truth_val) {
                            entries_matched++;
                            break;
                        }

                        // Calculate next chunk boundary
                        next_chunk_boundary = ((last_signaled[flat_idx] / tokens_per_chunk) + 1) * tokens_per_chunk;
                        if (next_chunk_boundary > ground_truth_val) {
                            next_chunk_boundary = ground_truth_val;
                        }
                    }
                } else {
                    // Non-drain core: detect when a new chunk value has arrived
                    if (current_val > last_signaled[flat_idx]) {
                        // Process all chunk boundaries between last_signaled and current_val
                        uint32_t next_chunk_boundary =
                            ((last_signaled[flat_idx] / tokens_per_chunk) + 1) * tokens_per_chunk;
                        if (next_chunk_boundary > ground_truth_val) {
                            next_chunk_boundary = ground_truth_val;
                        }

                        while (current_val >= next_chunk_boundary && last_signaled[flat_idx] < ground_truth_val) {
                            // CHUNK PROCESSING: Scan metadata to find tokens that selected this local_expert
                            process_chunk<
                                tile_height,
                                selected_experts_k,
                                experts_per_device,
                                tokens_per_device,
                                tilizer_input_cb_id,
                                aligned_output_page_size,
                                tokens_per_chunk>(
                                local_expert,
                                src_dev,
                                next_chunk_boundary,
                                linearized_mesh_coord,
                                indices_pages,
                                last_scanned_token[local_expert][src_dev],
                                tokens_found_count[local_expert][src_dev],
                                metadata_cb_addr,
                                metadata_tensor_addr_gen,
                                output_tensor_addr_gen,
                                devices_for_experts,
                                tilizer_subtoken_offset,
                                tilizer_subtoken_size);

                            // Update last_signaled
                            last_signaled[flat_idx] = next_chunk_boundary;
                            DPRINT << "E[" << local_expert << "][D" << src_dev << "] = " << next_chunk_boundary
                                   << ENDL();

                            // Check if we've reached ground truth
                            if (last_signaled[flat_idx] >= ground_truth_val) {
                                entries_matched++;
                                break;
                            }

                            // Calculate next chunk boundary
                            next_chunk_boundary = ((last_signaled[flat_idx] / tokens_per_chunk) + 1) * tokens_per_chunk;
                            if (next_chunk_boundary > ground_truth_val) {
                                next_chunk_boundary = ground_truth_val;
                            }
                        }
                    }
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
