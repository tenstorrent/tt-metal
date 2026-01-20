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

template <typename DataType>
FORCE_INLINE DataType* tile_row_offset(DataType* indices_address, uint32_t row) {
    constexpr uint32_t num_face_width = 2;
    constexpr uint32_t num_face_height = 2;
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t TileHeight = 32;
    constexpr uint32_t TileWidth = 32;
    uint32_t offset = 0;
    uint32_t local_row = row;
    if (row >= FaceHeight) {
        offset += num_face_width * FaceHeight * FaceWidth;  // if it was generic, multiply by row/FaceHeight
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
        offset += FaceHeight * FaceWidth;  // if it was generic, multiply by col/FaceWidth
        local_col -= FaceWidth;
    }
    offset += local_col;
    return (DataType*)(indices_address + offset);
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
    constexpr uint32_t tilizer_output_cb_id = get_named_compile_time_arg_val("tilizer_output_cb_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t total_chunks_cb_id = get_named_compile_time_arg_val("total_chunks_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t brisc_e_t_buffer_id = get_named_compile_time_arg_val("brisc_e_t_buffer_id");
    constexpr uint32_t brisc_expert_counts_cb_id = get_named_compile_time_arg_val("brisc_expert_counts_cb_id");
    constexpr uint32_t brisc_expert_activation_cb_id = get_named_compile_time_arg_val("brisc_expert_activation_cb_id");
    constexpr uint32_t brisc_activated_count_cb_id = get_named_compile_time_arg_val("brisc_activated_count_cb_id");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t e_t_entry_size = get_named_compile_time_arg_val("e_t_entry_size");

    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");
    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_scores_page_size = get_named_compile_time_arg_val("aligned_scores_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t hidden_size = get_named_compile_time_arg_val("hidden_size");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    // For parallel metadata processing - BRISC processes second half of tokens
    constexpr uint32_t brisc_token_start = tokens / 2;
    constexpr uint32_t brisc_token_end = tokens;
    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t tokens_per_device = tokens / dispatch_devices;

    // Tile dimensions
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t tiles_per_row = hidden_size / TILE_WIDTH;  // Number of tiles in width dimension

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    [[maybe_unused]] uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);     // 0
    [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 1
    [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    [[maybe_unused]] uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 3
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);                     // 4
    [[maybe_unused]] bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 5
    uint32_t tilizer_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);                   // 6
    uint32_t tilizer_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                     // 7

    // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);

    // Compute tiles_per_chunk for this core based on its subtoken portion
    // tile_width_bytes = TILE_WIDTH * element_size (element_size = output_page_size / (TILE_HEIGHT * TILE_WIDTH))
    constexpr uint32_t element_size = output_page_size / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t tile_width_bytes = TILE_WIDTH * element_size;
    uint32_t tiles_per_chunk = tilizer_subtoken_size / tile_width_bytes;

    // Compute width tile offset for this core
    uint32_t width_tile_start = tilizer_subtoken_offset / tile_width_bytes;

    // ========== BRISC PARALLEL METADATA PROCESSING ==========
    // BRISC processes second half of tokens (tokens/2 to tokens) in parallel with NCRISC
    // Aligned row size for expert_activation buffer (in bytes)
    constexpr uint32_t aligned_activation_row_bytes =
        ((2 * experts_per_device + 1) * sizeof(uint32_t) + l1_alignment - 1) / l1_alignment * l1_alignment;

    if (is_drain_tilizer_core) {
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
        cb_reserve_back(brisc_e_t_buffer_id, 1);
        const uint32_t brisc_e_t_buffer_base = get_write_ptr(brisc_e_t_buffer_id);
        constexpr uint32_t brisc_tokens_capacity = tokens / 2;  // Max tokens per expert for BRISC

        // Reserve BRISC's expert_activation buffer (single page contains all activation rows)
        cb_reserve_back(brisc_expert_activation_cb_id, 1);
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

        // Indices and scores are sharded on drain core, accessible via CB
        const uint32_t indices_base = get_read_ptr(indices_tensor_cb_id);
        const uint32_t scores_base = get_read_ptr(scores_tensor_cb_id);

        // Per-expert token counts for BRISC's half
        uint32_t brisc_num_tokens_per_expert[experts_per_device] = {0};
        uint32_t brisc_num_activated_tokens = 0;

        // Cache source_device_mapping - only changes every tokens_per_device tokens
        uint32_t prev_device_in_group = UINT32_MAX;
        const uint16_t* source_device_mapping = nullptr;

        // Process BRISC's token range: [brisc_token_start, brisc_token_end)
        for (uint32_t t = brisc_token_start; t < brisc_token_end; t++) {
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

            const uint16_t* token_indices =
                reinterpret_cast<const uint16_t*>(indices_base + t * aligned_indices_page_size);
            const uint16_t* token_scores =
                reinterpret_cast<const uint16_t*>(scores_base + t * aligned_scores_page_size);

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
                                brisc_expert_activation_base +
                                brisc_num_activated_tokens * aligned_activation_row_bytes);
                            brisc_activation_l1_ptr[0] = t;
                            activated = true;
                        }

                        DPRINT << "Token " << t << " activated expert " << selected_expert << " with k-index " << k
                               << ENDL();
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
        cb_push_back(brisc_e_t_buffer_id, 1);

        // Push BRISC's expert_activation buffer
        cb_push_back(brisc_expert_activation_cb_id, 1);

        // Push BRISC's per-expert counts to CB for NCRISC to read
        cb_reserve_back(brisc_expert_counts_cb_id, 1);
        uint32_t* brisc_counts_ptr = reinterpret_cast<uint32_t*>(get_write_ptr(brisc_expert_counts_cb_id));
        for (uint32_t e = 0; e < experts_per_device; e++) {
            brisc_counts_ptr[e] = brisc_num_tokens_per_expert[e];
        }
        cb_push_back(brisc_expert_counts_cb_id, 1);

        // Push BRISC's activated token count
        cb_reserve_back(brisc_activated_count_cb_id, 1);
        *reinterpret_cast<uint32_t*>(get_write_ptr(brisc_activated_count_cb_id)) = brisc_num_activated_tokens;
        cb_push_back(brisc_activated_count_cb_id, 1);
    }

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
    cb_wait_front(total_chunks_cb_id, 1);
    [[maybe_unused]] uint32_t total_chunks =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(total_chunks_cb_id));

    // Process each expert's chunks
    // Order matches reader: all chunks for expert 0, then expert 1, etc.
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t num_expert_tokens = num_tokens_per_expert[e];
        uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++) {
            // Wait for compute to push tiles_per_chunk tiles
            cb_wait_front(tilizer_output_cb_id, tiles_per_chunk);

            // Compute the tile row for this chunk
            // linear_row_start = expert * tokens + chunk * tokens_per_chunk
            // But for output, we use total_tokens (max possible), not actual activated tokens
            // Actually, for correctness, we write at the position based on actual token index
            uint32_t token_row_start = e * tokens + chunk * tokens_per_chunk;
            uint32_t tile_row = token_row_start / TILE_HEIGHT;

            // Get L1 read pointer for the tilized output
            uint32_t l1_read_addr = get_read_ptr(tilizer_output_cb_id);

            // Write each tile to DRAM
            for (uint32_t t = 0; t < tiles_per_chunk; t++) {
                uint32_t tile_col = width_tile_start + t;
                uint32_t tile_idx = tile_row * tiles_per_row + tile_col;

                uint64_t dst_noc_addr = get_noc_addr(tile_idx, output_tensor_addr_gen);
                uint32_t src_l1_addr = l1_read_addr + t * output_page_size;

                noc_async_write(src_l1_addr, dst_noc_addr, output_page_size);
            }
            noc_async_write_barrier();

            // Pop the tiles from CB
            cb_pop_front(tilizer_output_cb_id, tiles_per_chunk);
        }
    }

    // Pop the per-expert counts and total_chunks (cleanup)
    cb_pop_front(per_expert_total_tokens_cb_id, experts_per_device);
    cb_pop_front(total_chunks_cb_id, 1);
}
