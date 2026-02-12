// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
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

FORCE_INLINE void noc_async_write_linked_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr_multicast, uint32_t size, uint32_t num_dests, uint8_t noc) {
    while (size > NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast(src_local_l1_addr, dst_noc_addr_multicast, NOC_MAX_BURST_SIZE, num_dests, true, noc);
        src_local_l1_addr += NOC_MAX_BURST_SIZE;
        dst_noc_addr_multicast += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
    }
    noc_async_write_multicast(src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, false, noc);
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

template <
    uint32_t LinearizedMeshCoord,
    uint32_t TokensPerDevice,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup Axis>
inline uint32_t get_device_idx_from_global_token_idx(const uint32_t t) {
    constexpr uint32_t Replicate_Group = (Axis == ReplicateGroup::NONE)   ? MeshRows * MeshCols
                                         : (Axis == ReplicateGroup::COLS) ? MeshRows
                                                                          : MeshCols;
    const uint32_t device_in_group = t / TokensPerDevice;

    if constexpr (Axis == ReplicateGroup::NONE) {
        return device_in_group;
    } else if (Axis == ReplicateGroup::ROWS) {
        return (LinearizedMeshCoord / MeshCols) * MeshCols + device_in_group;
    } else {
        return device_in_group * MeshCols + LinearizedMeshCoord % MeshCols;
    }
}

void kernel_main() {
    // Compile-time arguments

    // CBs
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t total_chunks_cb_id = get_named_compile_time_arg_val("total_chunks_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t brisc_e_t_cb_id = get_named_compile_time_arg_val("brisc_e_t_cb_id");
    constexpr uint32_t brisc_expert_counts_cb_id = get_named_compile_time_arg_val("brisc_expert_counts_cb_id");
    constexpr uint32_t brisc_expert_activation_cb_id = get_named_compile_time_arg_val("brisc_expert_activation_cb_id");
    constexpr uint32_t brisc_activated_count_cb_id = get_named_compile_time_arg_val("brisc_activated_count_cb_id");

    // Alignment
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t e_t_entry_size = get_named_compile_time_arg_val("e_t_entry_size");

    // Number of pages
    constexpr uint32_t shared_cb_num_pages = get_named_compile_time_arg_val("shared_cb_num_pages");

    // Page sizes
    constexpr uint32_t tilize_output_page_size = get_named_compile_time_arg_val("tilize_output_page_size");

    // Aligned page sizes
    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_scores_page_size = get_named_compile_time_arg_val("aligned_scores_page_size");

    // General info
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t hidden_size = get_named_compile_time_arg_val("hidden_size");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");

    // Chunk info
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    // Mesh
    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    // Multicast coordinates for drain tilize to non-drain tilize synchronization
    constexpr uint32_t drain_core_noc_x = get_named_compile_time_arg_val("drain_core_noc_x");
    constexpr uint32_t drain_core_noc_y = get_named_compile_time_arg_val("drain_core_noc_y");

    // Gather groups
    constexpr uint32_t primary_mcast_gather_group_num_cores =
        get_named_compile_time_arg_val("primary_mcast_gather_group_num_cores");
    constexpr uint32_t secondary_mcast_gather_group_num_cores =
        get_named_compile_time_arg_val("secondary_mcast_gather_group_num_cores");

    // T multicast coordinates
    constexpr uint32_t num_tilize_cores = get_named_compile_time_arg_val("num_tilize_cores");

    constexpr uint32_t tilize_mcast_start_x = get_named_compile_time_arg_val("tilize_mcast_start_x");
    constexpr uint32_t tilize_mcast_start_y = get_named_compile_time_arg_val("tilize_mcast_start_y");
    constexpr uint32_t tilize_mcast_end_x = get_named_compile_time_arg_val("tilize_mcast_end_x");
    constexpr uint32_t tilize_mcast_end_y = get_named_compile_time_arg_val("tilize_mcast_end_y");
    constexpr uint32_t tilize_bounding_box_num_cores = get_named_compile_time_arg_val("tilize_bounding_box_num_cores");

    // Multicast coordinates for signalling MM cores
    constexpr uint32_t num_matmul_cores = get_named_compile_time_arg_val("num_matmul_cores");

    constexpr uint32_t matmul_mcast_start_x = get_named_compile_time_arg_val("matmul_mcast_start_x");
    constexpr uint32_t matmul_mcast_start_y = get_named_compile_time_arg_val("matmul_mcast_start_y");
    constexpr uint32_t matmul_mcast_end_x = get_named_compile_time_arg_val("matmul_mcast_end_x");
    constexpr uint32_t matmul_mcast_end_y = get_named_compile_time_arg_val("matmul_mcast_end_y");
    constexpr uint32_t matmul_bounding_box_num_cores = get_named_compile_time_arg_val("matmul_bounding_box_num_cores");

    // Semaphores
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
    constexpr uint32_t tilize_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("tilize_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    constexpr uint32_t previous_chunk_sent_semaphore_id =
        get_named_compile_time_arg_val("previous_chunk_sent_semaphore_id");

    uint32_t matmul_chunk_available_semaphore_addr = get_semaphore(matmul_chunk_available_semaphore_id);
    uint32_t tilize_chunk_ready_semaphore_addr = get_semaphore(tilize_chunk_ready_semaphore_id);
    uint32_t matmul_chunk_ready_semaphore_addr = get_semaphore(matmul_chunk_ready_semaphore_id);
    uint32_t previous_chunk_sent_semaphore_addr = get_semaphore(previous_chunk_sent_semaphore_id);

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    [[maybe_unused]] uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 0 - not used by writer
    [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 1 - not used by writer
    [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 2 - not used by writer
    [[maybe_unused]] uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 3 - not used by writer
    [[maybe_unused]] uint32_t per_expert_total_tokens_output_tensor_address =
        get_arg_val<uint32_t>(rt_args_idx++);  // 4 not used by writer
    [[maybe_unused]] uint32_t expert_activation_output_address =
        get_arg_val<uint32_t>(rt_args_idx++);                                                 // 5 - not used by writer
    [[maybe_unused]] uint32_t e_t_output_address = get_arg_val<uint32_t>(rt_args_idx++);      // 6 - not used by writer
    bool is_drain_tilize_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);                   // 7
    bool is_secondary_mcaster = (bool)get_arg_val<uint32_t>(rt_args_idx++);                   // 8
    uint32_t initial_mcast_gather_core_nox_x = get_arg_val<uint32_t>(rt_args_idx++);          // 9
    uint32_t initial_mcast_gather_core_nox_y = get_arg_val<uint32_t>(rt_args_idx++);          // 10
    uint32_t global_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);                   // 11
    uint32_t mcast_group_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);              // 12
    uint32_t mcast_group_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                // 13
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                            // 14
    uint32_t core_token_start = get_arg_val<uint32_t>(rt_args_idx++);                         // 15
    uint32_t core_token_end = get_arg_val<uint32_t>(rt_args_idx++);                           // 16
    [[maybe_unused]] uint32_t tilize_core_idx = get_arg_val<uint32_t>(rt_args_idx++);         // 17 - not used by writer

    // Constants
    constexpr uint32_t one_page = 1;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;
    constexpr uint32_t element_size = tilize_output_page_size / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t tile_width_bytes = TILE_WIDTH * element_size;

    // For parallel metadata processing - BRISC processes second half of this core's token range
    // Note: These are computed at runtime based on core_token_start/end in Step 3
    constexpr uint32_t brisc_token_start = tokens / 2;
    constexpr uint32_t brisc_token_end = tokens;
    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t tokens_per_device = tokens / dispatch_devices;

    // Compute width tile offset for this core
    uint32_t global_tile_offset = global_subtoken_offset / tile_width_bytes;
    uint32_t mcast_group_tile_offset = mcast_group_subtoken_offset / tile_width_bytes;

    // Compute tiles_per_local_chunk for this core based on its subtoken portion
    uint32_t tiles_per_global_chunk = hidden_size / TILE_WIDTH;
    uint32_t tiles_per_local_chunk = subtoken_size / tile_width_bytes;
    uint32_t tiles_per_mcast_group_chunk = mcast_group_subtoken_size / tile_width_bytes;

    // ========== ALL CORES: BRISC PARALLEL METADATA PROCESSING ==========
    // BRISC processes second half of this core's token range in parallel with NCRISC
    // Aligned row size for expert_activation buffer (in bytes)
    constexpr uint32_t aligned_activation_row_bytes =
        ((2 * experts_per_device + 1) * sizeof(uint32_t) + l1_alignment - 1) / l1_alignment * l1_alignment;

    // Calculate BRISC's token range for this core
    uint32_t tokens_this_core = core_token_end - core_token_start;
    uint32_t brisc_token_start_runtime = core_token_start + tokens_this_core / 2;
    uint32_t brisc_token_end_runtime = core_token_end;
    uint32_t brisc_tokens_capacity = tokens_this_core / 2;

    // Wait for NCRISC to finish reading the mapping tensor
    cb_wait_front(mapping_tensor_cb_id, num_devices);

    // Get mapping base pointer (read by NCRISC)
    const uint32_t mapping_base = get_read_ptr(mapping_tensor_cb_id);

    // Build local_expert_ids array - experts that map to this device
    uint16_t* expert_to_device_map =
        reinterpret_cast<uint16_t*>(mapping_base + linearized_mesh_coord * aligned_mapping_page_size);
    uint16_t local_expert_ids[experts_per_device];
    uint32_t local_expert_count = 0;
    for (uint32_t i = 0; i < experts; i++) {
        uint16_t expert_mesh_coord = expert_to_device_map[i];
        if (expert_mesh_coord == linearized_mesh_coord) {
            if (local_expert_count < experts_per_device) {
                local_expert_ids[local_expert_count] = i;
                local_expert_count++;
            }
        }
    }

    // Reserve BRISC's e_t buffer (single page contains all experts' token lists)
    cb_reserve_back(brisc_e_t_cb_id, one_page);
    const uint32_t brisc_e_t_buffer_base = get_write_ptr(brisc_e_t_cb_id);

    // Reserve BRISC's expert_activation buffer (single page contains all activation rows)
    cb_reserve_back(brisc_expert_activation_cb_id, one_page);
    const uint32_t brisc_expert_activation_base = get_write_ptr(brisc_expert_activation_cb_id);

    // Initialize BRISC's expert_activation buffer with sentinel values (selected_experts_k)
    for (uint32_t row = 0; row < brisc_tokens_capacity; row++) {
        uint32_t* row_ptr =
            reinterpret_cast<uint32_t*>(brisc_expert_activation_base + row * aligned_activation_row_bytes);
        row_ptr[0] = 0;  // token_id placeholder
        for (uint32_t e = 0; e < experts_per_device; e++) {
            row_ptr[1 + e] = selected_experts_k;      // sentinel for k-index
            row_ptr[1 + experts_per_device + e] = 0;  // score placeholder
        }
    }

    // Indices and scores accessible via CB (drain has shard, non-drain read via NOC in reader)
    const uint32_t indices_base = get_read_ptr(indices_tensor_cb_id);
    const uint32_t scores_base = get_read_ptr(scores_tensor_cb_id);

    // Per-expert token counts for BRISC's half
    uint32_t brisc_num_tokens_per_expert[experts_per_device] = {0};
    uint32_t brisc_num_activated_tokens = 0;

    // Cache source_device_mapping - only changes every tokens_per_device tokens
    uint32_t prev_device_in_group = UINT32_MAX;
    const uint16_t* source_device_mapping = nullptr;

    // Process BRISC's token range: [brisc_token_start_runtime, brisc_token_end_runtime)
    for (uint32_t t = brisc_token_start_runtime; t < brisc_token_end_runtime; t++) {
        const uint32_t device_in_group = t / tokens_per_device;

        // Only update mapping pointer when device_in_group changes
        if (device_in_group != prev_device_in_group) {
            const uint32_t source_device = get_device_idx_from_global_token_idx<
                linearized_mesh_coord,
                tokens_per_device,
                mesh_rows,
                mesh_cols,
                axis>(t);
            source_device_mapping =
                reinterpret_cast<const uint16_t*>(mapping_base + source_device * aligned_mapping_page_size);
            prev_device_in_group = device_in_group;
        }

        const uint16_t* token_indices = reinterpret_cast<const uint16_t*>(indices_base + t * aligned_indices_page_size);
        const uint16_t* token_scores = reinterpret_cast<const uint16_t*>(scores_base + t * aligned_scores_page_size);

        // Track if this token is activated for any local expert
        uint32_t* brisc_activation_l1_ptr = nullptr;
        bool activated = false;

        for (uint32_t k = 0; k < selected_experts_k; k++) {
            const uint16_t selected_expert = token_indices[k];

            // Check if this expert maps to our device first
            if (source_device_mapping[selected_expert] != linearized_mesh_coord) {
                continue;
            }

            // Check if it's one of our local experts
            for (uint32_t e = 0; e < local_expert_count; e++) {
                if (selected_expert == local_expert_ids[e]) {
                    // First activation for this token - set up pointer and write token id
                    if (!activated) {
                        brisc_activation_l1_ptr = reinterpret_cast<uint32_t*>(
                            brisc_expert_activation_base + brisc_num_activated_tokens * aligned_activation_row_bytes);
                        brisc_activation_l1_ptr[0] = t;
                        activated = true;
                    }

                    // Write k-index and score for this expert
                    brisc_activation_l1_ptr[1 + e] = k;
                    brisc_activation_l1_ptr[1 + experts_per_device + e] = static_cast<uint32_t>(token_scores[k]);

                    // Write to BRISC's e_t buffer (16B aligned entries)
                    const uint32_t brisc_e_t_offset =
                        (e * brisc_tokens_capacity + brisc_num_tokens_per_expert[e]) * e_t_entry_size;
                    *reinterpret_cast<uint32_t*>(brisc_e_t_buffer_base + brisc_e_t_offset) = t;
                    brisc_num_tokens_per_expert[e]++;
                    break;
                }
            }
        }

        if (activated) {
            brisc_num_activated_tokens++;
        }
    }

    // Push BRISC's e_t buffer (no -1 cap needed, NCRISC will cap final merged buffer)
    cb_push_back(brisc_e_t_cb_id, one_page);

    // Push BRISC's expert_activation buffer
    cb_push_back(brisc_expert_activation_cb_id, one_page);

    // Push BRISC's per-expert counts to CB for NCRISC to read
    cb_reserve_back(brisc_expert_counts_cb_id, one_page);
    uint32_t* brisc_counts_ptr = reinterpret_cast<uint32_t*>(get_write_ptr(brisc_expert_counts_cb_id));
    for (uint32_t e = 0; e < experts_per_device; e++) {
        brisc_counts_ptr[e] = brisc_num_tokens_per_expert[e];
    }
    cb_push_back(brisc_expert_counts_cb_id, one_page);

    // Push BRISC's activated token count
    cb_reserve_back(brisc_activated_count_cb_id, one_page);
    *reinterpret_cast<uint32_t*>(get_write_ptr(brisc_activated_count_cb_id)) = brisc_num_activated_tokens;
    cb_push_back(brisc_activated_count_cb_id, one_page);

    // Wait for reader to push per-expert token counts (includes merged NCRISC + BRISC counts)
    cb_wait_front(per_expert_total_tokens_cb_id, 1);
    volatile tt_l1_ptr uint32_t* per_expert_counts =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(per_expert_total_tokens_cb_id));

    // Read per-expert token counts into local array
    uint32_t num_tokens_per_expert[experts_per_device];
    for (uint32_t e = 0; e < experts_per_device; e++) {
        num_tokens_per_expert[e] = per_expert_counts[e];
    }

    // Wait for reader to push total_chunks
    cb_wait_front(total_chunks_cb_id, one_page);
    [[maybe_unused]] uint32_t total_chunks =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(total_chunks_cb_id));

    /************************************************************************/
    /* Synchronization setup for signalling between tilize and matmul cores */
    /************************************************************************/

    // Semaphore all tilize cores wait on to indicate we cand send another chunk
    // Matmul sends to tilize drain sync core, which propagates it to the tilize non-drain-sync cores
    uint32_t matmul_chunk_available_semaphore_wait_value = num_matmul_cores;
    uint64_t matmul_chunk_available_semaphore_tilize_mcast_addr = get_safe_multicast_noc_addr(
        tilize_mcast_start_x,
        tilize_mcast_start_y,
        tilize_mcast_end_x,
        tilize_mcast_end_y,
        matmul_chunk_available_semaphore_addr,
        noc_index);

    // Semaphore we use to signal to matmul cores that a chunk has arrived
    uint32_t matmul_chunk_ready_semaphore_set_value = 1;
    uint64_t matmul_chunk_ready_semaphore_mcast_addr = get_safe_multicast_noc_addr(
        matmul_mcast_start_x,
        matmul_mcast_start_y,
        matmul_mcast_end_x,
        matmul_mcast_end_y,
        matmul_chunk_ready_semaphore_addr,
        noc_index);

    // How many chunks we've sent to matmul so far
    uint32_t num_chunks_sent = 0;

    // This decides which half of the matmul input buffer we send to
    bool use_second_half_buffer = false;
    uint32_t matmul_chunk_input_cb_base_addr = get_read_ptr(tilize_output_cb_id);
    uint32_t first_half_buffer_addr = matmul_chunk_input_cb_base_addr;
    uint32_t second_half_buffer_addr =
        matmul_chunk_input_cb_base_addr + (tiles_per_global_chunk * tilize_output_page_size);

    // For synchronization between the drain-sync core and non-drain-sync cores
    uint32_t tilize_chunk_ready_wait_value = num_tilize_cores - 1;
    uint64_t tilize_chunk_ready_drain_semaphore_noc_addr =
        get_noc_addr(drain_core_noc_x, drain_core_noc_y, tilize_chunk_ready_semaphore_addr, noc_index);
    uint64_t tilze_chunk_ready_mcast_addr = get_safe_multicast_noc_addr(
        tilize_mcast_start_x,
        tilize_mcast_start_y,
        tilize_mcast_end_x,
        tilize_mcast_end_y,
        tilize_chunk_ready_semaphore_addr,
        noc_index);

    // mcast address for the first half of the buffer
    uint64_t first_half_buffer_matmul_chunk_input_mcast_addr = get_safe_multicast_noc_addr(
        matmul_mcast_start_x,
        matmul_mcast_start_y,
        matmul_mcast_end_x,
        matmul_mcast_end_y,
        first_half_buffer_addr,
        noc_index);

    // mcast address for the second half of the buffer
    uint64_t second_half_buffer_matmul_chunk_input_mcast_addr = get_safe_multicast_noc_addr(
        matmul_mcast_start_x,
        matmul_mcast_start_y,
        matmul_mcast_end_x,
        matmul_mcast_end_y,
        second_half_buffer_addr,
        noc_index);

    // how many bytes the single mcaster sends on each subsequent iteration
    uint32_t normal_iteration_bytes_to_mcast = tiles_per_global_chunk * tilize_output_page_size;

    /* start loop iterations */

    // Process each expert's chunks
    // Order matches reader: all chunks for expert 0, then expert 1, etc.
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t num_expert_tokens = num_tokens_per_expert[e];
        uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++) {
            // Wait for compute to push tiles_per_local_chunk tiles
            cb_wait_front(tilize_output_cb_id, shared_cb_num_pages);
            uint32_t l1_read_addr = get_read_ptr(tilize_output_cb_id);

            /*
             * Send chunks to MM cores;
             * 1) all tilize cores wait for buffer on matmul cores to be free
             *    - can skip for the first two chunks
             * 2) Start process of sending the chunk to the MM cores, first gather the sub-chunks on certain tilize
             * cores
             *    - First Iteration: 2 mcasters, one per NoC
             *    - All Subsequent Iterations: 1 mcaster using NoC1, leaving NoC0 for MM to read in weights
             *    - If we only have a single tilize core, ignore the special First Iteration scheme
             * First Iteration:
             *   3a) non mcaster cores send their sub-chunk to their designated mcaster
             *   4a) non mcaster cores signal to their designated mcaster that they've sent their chunk
             *   5a) mcaster cores wait until all sub-chunks are gathered
             *   6a) mcasters mcast gathered sub-chunks to MM cores
             *   7a) mcaster B (non-drain-sync mcaster) barriers to let mcast complete (on NoC0), then signals to
             *       mcaster A (drain-sync mcaster) that the B sub-chunks have been delivered
             *   8a) mcaster A waits for signal from mcaster B that all B sub-chunks have been delivered
             *   NOTE: mcaster A is the drain-sync core
             * All Subsequent Iterations:
             *   3b) non-drain-sync cores send their sub-chunk to the T drain-sync core
             *   4b) non-drain-sync cores signal to the T drain-sync core that they've sent their chunk
             *   5b) drain-sync core waits until all chunks are gathered
             *   6b) drain-sync core mcasts gathered chunk to MM cores
             * 9) drain-sync core (mcaster A) mcasts to MM cores that the chunk has arrived
             * 10) drain-sync core signals to non-drain-sync cores that all chunks have been sent
             * 11) non-drain-sync cores wait for signal from drain-sync-core that chunks have been sent
             * 12) all tilize writers signal to their reader that they can begin reading in tokens again
             *     pop their tilize input CB (allowing readers to read in another set of tokens)
             *
             * NOTE: Steps 10-12 ensure we don't use NoC1 to read in another set of tokens, or mcast another chunk
             *       of tilized tokens, or write to an output tensor while we're still mcasting the previous chunk.
             *       This ensures there's no NoC contention during the mcast phase (which is also a requirement in
             *       order to use linked mcasts)
             */

            // == 1 ==
            // skip for the first two chunks (2 chunks are allocated, and both are initially empty)
            if (num_chunks_sent >= 2) {
                // wait_min as MM may signal twice (once per buffer slot) before we acknowledge the first (empty) buffer
                // slot
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_available_semaphore_addr),
                    matmul_chunk_available_semaphore_wait_value);
                matmul_chunk_available_semaphore_wait_value += num_matmul_cores;

                // MM only signals to the drain-sync-core, which then forwards it to the non-drain-sync cores
                if (is_drain_tilize_core && num_tilize_cores > 1) {
                    // use the local value of the semaphore, which is the value we just waited on (no need to set local
                    // value)
                    noc_semaphore_set_multicast(
                        matmul_chunk_available_semaphore_addr,
                        matmul_chunk_available_semaphore_tilize_mcast_addr,
                        tilize_bounding_box_num_cores - 1,
                        false,
                        noc_index);
                }
            }

            // == 2 ==
            // start send sequence
            if (num_chunks_sent == 0 && num_tilize_cores > 1) {
                // FIRST ITERATION
                // NOTE: the first tilize iteration always sends to the first half of the matmul input buffer

                uint32_t bytes_to_mcast = tiles_per_mcast_group_chunk * tilize_output_page_size;
                if (is_drain_tilize_core) {
                    // mcasts data over NoC1 (primary NoC for tilize cores)

                    // == 5a ==
                    // wait for all gathered sub-chunks to arrive
                    noc_semaphore_wait(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                        primary_mcast_gather_group_num_cores - 1);

                    // == 6a ==

                    // mcast address for the primary mcast group, sends over NoC1, to the first half of the chunk (of
                    // the first half of the buffer)
                    uint64_t first_iteration_primary_mcast_group_matmul_chunk_input_mcast_addr =
                        get_safe_multicast_noc_addr(
                            matmul_mcast_start_x,
                            matmul_mcast_start_y,
                            matmul_mcast_end_x,
                            matmul_mcast_end_y,
                            first_half_buffer_addr,
                            noc_index);

                    // mcast the data
                    noc_async_write_linked_multicast(
                        l1_read_addr,
                        first_iteration_primary_mcast_group_matmul_chunk_input_mcast_addr,
                        bytes_to_mcast,
                        matmul_bounding_box_num_cores,
                        noc_index);

                    // == 8a ==
                    // wait for secondary mcaster to signal that all secondary mcast sub-chunks have been delivered
                    noc_semaphore_wait(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                        tilize_chunk_ready_wait_value);
                    tilize_chunk_ready_wait_value += (num_tilize_cores - 1);
                } else if (is_secondary_mcaster) {
                    // mcasts data over NoC0 (secondary NoC for tilize cores, usually reserved for MM cores reading
                    // weights from DRAM)

                    // == 5a ==
                    // wait for all gathered sub-chunks to arrive
                    noc_semaphore_wait(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                        secondary_mcast_gather_group_num_cores - 1);

                    // == 6a ==

                    // mcast address for the secondary mcast group, sends over NoC0, to the second half of the chunk (of
                    // the first half of the buffer)
                    uint64_t first_iteration_secondary_mcast_group_matmul_chunk_input_mcast_addr =
                        get_safe_multicast_noc_addr(
                            matmul_mcast_start_x,
                            matmul_mcast_start_y,
                            matmul_mcast_end_x,
                            matmul_mcast_end_y,
                            first_half_buffer_addr + global_tile_offset * tilize_output_page_size,
                            1 - noc_index);

                    // mcast the data
                    noc_async_write_linked_multicast(
                        l1_read_addr,
                        first_iteration_secondary_mcast_group_matmul_chunk_input_mcast_addr,
                        bytes_to_mcast,
                        matmul_bounding_box_num_cores,
                        1 - noc_index);

                    // == 7a ==
                    // explicit barrier since we're on different NoC than primary mcaster (which is the one that signals
                    // to MM cores) signal to primary mcaster (drain-sync core) that we've sent our sub-chunks
                    noc_async_write_barrier(1 - noc_index);
                    noc_semaphore_inc(
                        tilize_chunk_ready_drain_semaphore_noc_addr, secondary_mcast_gather_group_num_cores, noc_index);
                } else {
                    // initial_gather_noc

                    // == 3a ==
                    // send to proper offset on our initial mcast gather core
                    uint32_t target_l1_initial_gather_addr =
                        l1_read_addr + mcast_group_tile_offset * tilize_output_page_size;
                    uint32_t initial_gather_noc_addr = get_noc_addr(
                        initial_mcast_gather_core_nox_x,
                        initial_mcast_gather_core_nox_y,
                        target_l1_initial_gather_addr,
                        noc_index);
                    noc_async_write(
                        l1_read_addr,
                        initial_gather_noc_addr,
                        tiles_per_local_chunk * tilize_output_page_size,
                        noc_index);

                    // == 4a ==
                    // signal to our initial mcast gather core that we've delivered our sub-chunk
                    uint64_t initial_gather_semaphore_noc_addr = get_noc_addr(
                        initial_mcast_gather_core_nox_x,
                        initial_mcast_gather_core_nox_y,
                        tilize_chunk_ready_semaphore_addr,
                        noc_index);
                    noc_semaphore_inc(initial_gather_semaphore_noc_addr, 1, noc_index);
                }
            } else {
                // ALL SUBSEQUENT ITERATIONS
                // mcasts data over NoC1 (primary NoC for tilize cores)

                if (is_drain_tilize_core) {
                    // == 5b ==
                    // wait until non-drain-sync cores send us their sub-chunks
                    noc_semaphore_wait(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                        tilize_chunk_ready_wait_value);
                    tilize_chunk_ready_wait_value += (num_tilize_cores - 1);

                    // == 6b ==

                    // which half of the input buffer we're sending to
                    uint64_t matmul_chunk_input_mcast_addr = use_second_half_buffer
                                                                 ? second_half_buffer_matmul_chunk_input_mcast_addr
                                                                 : first_half_buffer_matmul_chunk_input_mcast_addr;

                    // mcast the data
                    noc_async_write_linked_multicast(
                        l1_read_addr,
                        matmul_chunk_input_mcast_addr,
                        normal_iteration_bytes_to_mcast,
                        matmul_bounding_box_num_cores,
                        noc_index);
                } else {
                    // == 3b ==
                    // send to proper offset on global mmcast gather core (the drain-sync core)
                    uint32_t gather_addr = l1_read_addr + global_tile_offset * tilize_output_page_size;
                    uint32_t drain_gather_noc_addr =
                        get_noc_addr(drain_core_noc_x, drain_core_noc_y, gather_addr, noc_index);
                    noc_async_write(
                        l1_read_addr,
                        drain_gather_noc_addr,
                        tiles_per_local_chunk * tilize_output_page_size,
                        noc_index);

                    // == 4b ==
                    // signal to global mcast gather core that we've delivered our sub-chunk
                    noc_semaphore_inc(tilize_chunk_ready_drain_semaphore_noc_addr, 1, noc_index);
                }
            }

            if (is_drain_tilize_core) {
                // == 9 ==
                // signal to MM cores that entire chunk has arrived

                // set local value
                noc_semaphore_set(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_ready_semaphore_addr),
                    matmul_chunk_ready_semaphore_set_value);
                matmul_chunk_ready_semaphore_set_value++;

                // mcast sem set
                noc_semaphore_set_multicast(
                    matmul_chunk_ready_semaphore_addr,
                    matmul_chunk_ready_semaphore_mcast_addr,
                    matmul_bounding_box_num_cores,
                    false,
                    noc_index);

                // == 10 ==
                if (num_tilize_cores > 1) {
                    // Signal to non-drain-sync cores that they can start sending the next chunk
                    // Use local semaphore value (no need to explicitly set it)
                    // Local value is from when drain-sync waits until gather process is done (8a and 5b)
                    noc_semaphore_set_multicast(
                        tilize_chunk_ready_semaphore_addr,
                        tilze_chunk_ready_mcast_addr,
                        tilize_bounding_box_num_cores - 1,
                        false,
                        noc_index);
                }
            } else {
                // == 11 ==
                // wait until drain-sync signals to us that we can start using NoC again (read in next set of tokens,
                // output another chunk, etc)
                noc_semaphore_wait(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                    tilize_chunk_ready_wait_value);
                tilize_chunk_ready_wait_value += (num_tilize_cores - 1);
            }

            // we already barrier when using (1 - noc_index), so just need to flush on noc_index here
            noc_async_writes_flushed(noc_index);

            // pop the tiles from CB
            cb_pop_front(tilize_output_cb_id, shared_cb_num_pages);
            num_chunks_sent++;
            use_second_half_buffer = !use_second_half_buffer;

            // == 12 ==
            // signal to reader that they can start reading in another set of tokens
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(previous_chunk_sent_semaphore_addr), num_chunks_sent);
        }
    }

    // Pop the per-expert counts and total_chunks (cleanup)
    cb_pop_front(per_expert_total_tokens_cb_id, one_page);
    cb_pop_front(total_chunks_cb_id, one_page);

    noc_async_write_barrier(noc_index);
    noc_async_atomic_barrier(noc_index);
}
