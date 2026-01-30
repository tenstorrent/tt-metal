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
    constexpr uint32_t cb_s2c_in_id = get_named_compile_time_arg_val("cb_s2c_in_id");

    // Alignment
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t e_t_entry_size = get_named_compile_time_arg_val("e_t_entry_size");

    // Page sizes
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");

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

    // T multicast coordinates
    constexpr uint32_t tilize_mcast_start_x = get_named_compile_time_arg_val("tilize_mcast_start_x");
    constexpr uint32_t tilize_mcast_start_y = get_named_compile_time_arg_val("tilize_mcast_start_y");
    constexpr uint32_t tilize_mcast_end_x = get_named_compile_time_arg_val("tilize_mcast_end_x");
    constexpr uint32_t tilize_mcast_end_y = get_named_compile_time_arg_val("tilize_mcast_end_y");
    constexpr uint32_t num_tilize_cores = get_named_compile_time_arg_val("num_tilize_cores");

    // Multicast coordinates for signalling MM cores
    constexpr uint32_t matmul_mcast_start_x = get_named_compile_time_arg_val("matmul_mcast_start_x");
    constexpr uint32_t matmul_mcast_start_y = get_named_compile_time_arg_val("matmul_mcast_start_y");
    constexpr uint32_t matmul_mcast_end_x = get_named_compile_time_arg_val("matmul_mcast_end_x");
    constexpr uint32_t matmul_mcast_end_y = get_named_compile_time_arg_val("matmul_mcast_end_y");
    constexpr uint32_t num_matmul_cores = get_named_compile_time_arg_val("num_matmul_cores");
    constexpr uint32_t num_matmul_bounding_box_cores = get_named_compile_time_arg_val("num_matmul_bounding_box_cores");

    // Semaphores
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
    constexpr uint32_t e0_tilize_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("e0_tilize_chunk_ready_semaphore_id");
    constexpr uint32_t e1_tilize_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("e1_tilize_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");

    uint32_t matmul_chunk_available_semaphore_addr = get_semaphore(matmul_chunk_available_semaphore_id);
    uint32_t e0_tilize_chunk_ready_semaphore_addr = get_semaphore(e0_tilize_chunk_ready_semaphore_id);
    uint32_t e1_tilize_chunk_ready_semaphore_addr = get_semaphore(e1_tilize_chunk_ready_semaphore_id);
    uint32_t matmul_chunk_ready_semaphore_addr = get_semaphore(matmul_chunk_ready_semaphore_id);

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    [[maybe_unused]] uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 0 - not used by writer
    [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 1 - not used by writer
    [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 2 - not used by writer
    [[maybe_unused]] uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 3 - not used by writer
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);                    // 4
    [[maybe_unused]] uint32_t expert_activation_output_address =
        get_arg_val<uint32_t>(rt_args_idx++);                                                 // 5 - not used by writer
    [[maybe_unused]] uint32_t e_t_output_address = get_arg_val<uint32_t>(rt_args_idx++);      // 6 - not used by writer
    [[maybe_unused]] bool is_drain_tilize_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 7
    uint32_t tilize_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);                   // 8
    uint32_t tilize_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                     // 9
    uint32_t core_token_start = get_arg_val<uint32_t>(rt_args_idx++);                         // 10
    uint32_t core_token_end = get_arg_val<uint32_t>(rt_args_idx++);                           // 11
    [[maybe_unused]] uint32_t tilize_core_idx = get_arg_val<uint32_t>(rt_args_idx++);         // 12 - not used by writer

    // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output, expert_activation_output,
    // e_t_output
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    // expert_activation_output_args not needed by writer
    // e_t_output_args not needed by writer

    // TensorAccessors
    const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);

    // Constants
    constexpr uint32_t one_page = 1;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t tiles_per_row = hidden_size / TILE_WIDTH;  // Number of tiles in width dimension

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    // For parallel metadata processing - BRISC processes second half of this core's token range
    // Note: These are computed at runtime based on core_token_start/end in Step 3
    constexpr uint32_t brisc_token_start = tokens / 2;
    constexpr uint32_t brisc_token_end = tokens;
    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t tokens_per_device = tokens / dispatch_devices;

    // Compute tiles_per_chunk for this core based on its subtoken portion
    // tile_width_bytes = TILE_WIDTH * element_size (element_size = output_page_size / (TILE_HEIGHT * TILE_WIDTH))
    constexpr uint32_t element_size = output_page_size / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t tile_width_bytes = TILE_WIDTH * element_size;
    uint32_t tiles_per_chunk = tilize_subtoken_size / tile_width_bytes;

    // Compute width tile offset for this core
    uint32_t width_tile_start = tilize_subtoken_offset / tile_width_bytes;

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
    cb_reserve_back(brisc_expert_counts_cb_id, 1);
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
    cb_wait_front(per_expert_total_tokens_cb_id, experts_per_device);
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

    /* Synchronization info for signalling between tilize and matmul cores */

    // Semaphore we wait on to indicate we cand send another chunk
    uint32_t matmul_chunk_available_semaphore_wait_value = num_matmul_cores;
    uint64_t matmul_chunk_available_semaphore_mcast_addr = get_safe_multicast_noc_addr(
        tilize_mcast_start_x,
        tilize_mcast_start_y,
        tilize_mcast_end_x,
        tilize_mcast_end_y,
        matmul_chunk_available_semaphore_addr);

    // Base address on the matmul cores we send chunks to
    uint32_t matmul_chunk_input_cb_addr = get_read_ptr(cb_s2c_in_id);

    // Semaphore we use to signal to matmul cores that a chunk has arrived
    uint32_t matmul_chunk_ready_semaphore_set_value = 1;
    uint64_t matmul_chunk_ready_semaphore_mcast_addr = get_safe_multicast_noc_addr(
        matmul_mcast_start_x,
        matmul_mcast_start_y,
        matmul_mcast_end_x,
        matmul_mcast_end_y,
        matmul_chunk_ready_semaphore_addr);

    // Process each expert's chunks
    // Order matches reader: all chunks for expert 0, then expert 1, etc.
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t num_expert_tokens = num_tokens_per_expert[e];
        uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;

        /* Synchronization info for signalling between tilize and matmul cores */

        // Determine which semaphore we're using to coordinate with drain-sync core
        uint32_t tilize_chunk_ready_semaphore_addr =
            e == 0 ? e0_tilize_chunk_ready_semaphore_addr : e1_tilize_chunk_ready_semaphore_addr;

        // Value drain-sync core waits on from non-drain-sync cores before sending semaphore increment to matmul cores
        uint32_t tilize_chunk_ready_wait_value = num_tilize_cores - 1;

        // Address non-drain-sync cores send semaphore increment to
        uint64_t drain_semaphore_noc_addr =
            get_noc_addr(drain_core_noc_x, drain_core_noc_y, tilize_chunk_ready_semaphore_addr);

        // TODO: (GR) this may change
        // Address we send the current chunk to, each expert has one chunk reserved
        uint32_t matmul_chunk_input_l1_addr =
            matmul_chunk_input_cb_addr + (e * tiles_per_row + width_tile_start) * output_page_size;
        uint64_t matmul_chunk_input_mcast_addr = get_safe_multicast_noc_addr(
            matmul_mcast_start_x,
            matmul_mcast_start_y,
            matmul_mcast_end_x,
            matmul_mcast_end_y,
            matmul_chunk_input_l1_addr);

        for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++) {
            // Wait for compute to push tiles_per_chunk tiles
            cb_wait_front(tilize_output_cb_id, tiles_per_chunk);
            uint32_t l1_read_addr = get_read_ptr(tilize_output_cb_id);

            /* START - WRITE TO MATMUL */

            /*
             * Send chunks to MM cores;
             * 1) Wait for buffer on matmul cores to be free
             *    - Can skip if it's the first chunk per expert
             * 2) Send the chunk
             * 3) Signal via semaphore to T drain-sync-core that you've sent your chunk
             * 4) T drain-sync-core waits until all T non-drain-sync cores have sent their chunk
             * 5) T drain-sync-core signals to MM cores that chunks have arrived
             */

            // == 1 ==
            // can skip for the first chunk per expert (1 chunk allocated per expert)
            if (chunk != 0) {
                noc_semaphore_wait(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_available_semaphore_addr),
                    matmul_chunk_available_semaphore_wait_value);
                matmul_chunk_available_semaphore_wait_value += num_matmul_cores;

                // MM only signals to the T drain-sync-core, which then forwards it to the T non-drain-sync cores
                if (is_drain_tilize_core && num_tilize_cores > 1) {
                    // Uses the local value of the semaphore, which is the value we just waited on
                    noc_semaphore_set_multicast(
                        matmul_chunk_available_semaphore_addr,
                        matmul_chunk_available_semaphore_mcast_addr,
                        num_tilize_cores - 1);
                }
            }

            // == 2 ==
            noc_async_write_multicast(
                l1_read_addr,
                matmul_chunk_input_mcast_addr,
                tiles_per_chunk * output_page_size,
                num_matmul_bounding_box_cores);
            noc_async_write_barrier();

            if (is_drain_tilize_core) {
                if (num_tilize_cores > 1) {
                    // == 4 ==
                    noc_semaphore_wait(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilize_chunk_ready_semaphore_addr),
                        tilize_chunk_ready_wait_value);
                    tilize_chunk_ready_wait_value += (num_tilize_cores - 1);
                }

                // == 5 ==
                // set local value
                noc_semaphore_set(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_ready_semaphore_addr),
                    matmul_chunk_ready_semaphore_set_value);
                matmul_chunk_ready_semaphore_set_value++;

                noc_semaphore_set_multicast(
                    matmul_chunk_ready_semaphore_addr,
                    matmul_chunk_ready_semaphore_mcast_addr,
                    num_matmul_bounding_box_cores);
            } else {
                // == 3 ==
                // signal drain-sync core that we've sent our chunk
                noc_semaphore_inc(drain_semaphore_noc_addr, 1);
            }

            /* END - WRITE TO MATMUL */

            /* START - WRITE TO OUTPUT TENSOR */
            // TODO: (GR) remove this block during integration

            // uint32_t l1_read_addr = get_read_ptr(tilize_output_cb_id);

            // Compute the tile row for this chunk
            // linear_row_start = expert * tokens + chunk * tokens_per_chunk
            // But for output, we use total_tokens (max possible), not actual activated tokens
            // Actually, for correctness, we write at the position based on actual token index
            uint32_t token_row_start = e * tokens + chunk * tokens_per_chunk;
            uint32_t tile_row = token_row_start / TILE_HEIGHT;

            // Write each tile to DRAM
            for (uint32_t t = 0; t < tiles_per_chunk; t++) {
                uint32_t tile_col = width_tile_start + t;
                uint32_t tile_idx = tile_row * tiles_per_row + tile_col;

                uint64_t dst_noc_addr = get_noc_addr(tile_idx, output_tensor_addr_gen);
                uint32_t src_l1_addr = l1_read_addr + t * output_page_size;

                noc_async_write(src_l1_addr, dst_noc_addr, output_page_size);
            }
            noc_async_write_barrier();
            /* END - WRITE TO OUTPUT TENSOR */

            // Pop the tiles from CB
            cb_pop_front(tilize_output_cb_id, tiles_per_chunk);
        }
    }

    // Pop the per-expert counts and total_chunks (cleanup)
    cb_pop_front(per_expert_total_tokens_cb_id, experts_per_device);
    cb_pop_front(total_chunks_cb_id, one_page);

    noc_async_write_barrier();
}
