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

// Initialize the expert activation buffer with default values:
// Each row (one per token) has 2*experts_per_device + 1 uint32_t values:
//   [token_id=-1, expert_0_activated=k+1, ..., expert_E-1_activated=k+1, score_0=0, ..., score_E-1=0]
// k+1 indicates "not activated" (will be overwritten with k-index if activated)
// Rows are aligned to l1_alignment for efficient NOC transfers.
//
// Strategy: Exponential doubling for maximum NOC efficiency.
// 1. Write first 2 rows via L1 (~90 cycles)
// 2. Double rows (2→4→8→16...) with barriers until copy size >= 512B (max throughput)
// 3. Parallel dispatch remaining copies at max efficiency
// 4. Handle remainder rows
//
// Caller must call noc_async_read_barrier() before using the buffer.
template <uint32_t selected_experts_k, uint32_t tokens, uint32_t experts_per_device, uint32_t l1_alignment>
FORCE_INLINE void init_expert_activation_buffer_async(uint32_t cb_id) {
    constexpr uint32_t row_elements = 2 * experts_per_device + 1;
    constexpr uint32_t row_size_bytes_unaligned = row_elements * sizeof(uint32_t);
    // Align row size to l1_alignment for NOC transfer efficiency
    constexpr uint32_t aligned_row_size_bytes =
        ((row_size_bytes_unaligned + l1_alignment - 1) / l1_alignment) * l1_alignment;

    // Minimum transfer size for maximum NOC throughput (~27.8 bytes/cycle)
    constexpr uint32_t MIN_EFFICIENT_BYTES = 512;

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

    // Write first row manually via L1 (~45 cycles for E=2)
    buffer[0] = static_cast<uint32_t>(-1);  // token_id = -1
    for (uint32_t e = 0; e < experts_per_device; e++) {
        buffer[1 + e] = selected_experts_k + 1;  // not activated
    }
    for (uint32_t e = 0; e < experts_per_device; e++) {
        buffer[1 + experts_per_device + e] = 0;  // score = 0
    }

    if constexpr (tokens <= 1) {
        return;  // Only 1 token, nothing more to do
    }

    uint64_t src_noc_addr = get_noc_addr(l1_write_addr);
    uint32_t rows_filled = 1;

    // Phase 1: Exponential doubling (1→2→4→8→16...) until copy size >= MIN_EFFICIENT_BYTES
    while (rows_filled < tokens) {
        uint32_t copy_size_bytes = rows_filled * aligned_row_size_bytes;

        if (copy_size_bytes >= MIN_EFFICIENT_BYTES) {
            // We've reached efficient size, break to parallel dispatch phase
            break;
        }

        // How many rows we can copy (limited by remaining space)
        uint32_t rows_to_copy = (rows_filled + rows_filled <= tokens) ? rows_filled : (tokens - rows_filled);

        if (rows_to_copy == 0) {
            break;
        }

        // Copy rows from start to current position
        uint32_t dest_addr = l1_write_addr + rows_filled * aligned_row_size_bytes;
        uint32_t bytes_to_copy = rows_to_copy * aligned_row_size_bytes;
        noc_async_read(src_noc_addr, dest_addr, bytes_to_copy);
        noc_async_read_barrier();  // Must barrier before next doubling iteration

        rows_filled += rows_to_copy;
    }

    // Phase 2: Parallel dispatch remaining copies at max efficiency (no barriers between)
    if (rows_filled < tokens) {
        uint32_t chunk_rows = rows_filled;  // Copy this many rows at a time
        uint32_t chunk_bytes = chunk_rows * aligned_row_size_bytes;

        // Dispatch all full chunks in parallel
        while (rows_filled + chunk_rows <= tokens) {
            uint32_t dest_addr = l1_write_addr + rows_filled * aligned_row_size_bytes;
            noc_async_read(src_noc_addr, dest_addr, chunk_bytes);
            rows_filled += chunk_rows;
        }

        // Handle remainder rows (if tokens not divisible by chunk_rows)
        if (rows_filled < tokens) {
            uint32_t remainder_rows = tokens - rows_filled;
            uint32_t remainder_bytes = remainder_rows * aligned_row_size_bytes;
            uint32_t dest_addr = l1_write_addr + rows_filled * aligned_row_size_bytes;
            noc_async_read(src_noc_addr, dest_addr, remainder_bytes);
        }
    }

    // Note: Caller must call noc_async_read_barrier() before using the buffer,
    // then cb_push_back(cb_id, tokens) when ready to make data available.
}

// Debug print function for expert_activation buffer
// Prints each row showing: [token_id | expert_activations... | scores...]
// start_token/end_token allow printing a subset of rows
template <uint32_t experts_per_device, uint32_t l1_alignment>
FORCE_INLINE void print_expert_activation_buffer(
    uint32_t cb_id, uint32_t start_token = 0, uint32_t end_token = 0xFFFFFFFF) {
    constexpr uint32_t row_elements = 2 * experts_per_device + 1;
    constexpr uint32_t row_size_bytes_unaligned = row_elements * sizeof(uint32_t);
    constexpr uint32_t aligned_row_size_bytes =
        ((row_size_bytes_unaligned + l1_alignment - 1) / l1_alignment) * l1_alignment;
    constexpr uint32_t aligned_row_elements = aligned_row_size_bytes / sizeof(uint32_t);

    volatile tt_l1_ptr uint32_t* buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_id));

    DPRINT << "=== Expert Activation Buffer ===" << ENDL();
    DPRINT << "Row format: [token_id | act_0..act_" << (experts_per_device - 1) << " | score_0..score_"
           << (experts_per_device - 1) << "]" << ENDL();

    for (uint32_t t = start_token; t < end_token; t++) {
        uint32_t base = t * aligned_row_elements;

        // Token ID (stored as uint32_t, but -1 means unset)
        uint32_t token_id = buffer[base];
        DPRINT << "T" << t << ": [";
        if (token_id == static_cast<uint32_t>(-1)) {
            DPRINT << "-1";
        } else {
            DPRINT << token_id;
        }
        DPRINT << " |";

        // Expert activations (k+1 means not activated, 0..k-1 means activated with that k-index)
        for (uint32_t e = 0; e < experts_per_device; e++) {
            DPRINT << " " << buffer[base + 1 + e];
        }
        DPRINT << " |";

        // Scores
        for (uint32_t e = 0; e < experts_per_device; e++) {
            DPRINT << " " << BF16(static_cast<uint16_t>(buffer[base + 1 + experts_per_device + e]));
        }
        DPRINT << "]" << ENDL();
    }
    DPRINT << "================================" << ENDL();
}

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

void kernel_main() {
    DPRINT << "Reader tilizer core started" << ENDL();
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t tilizer_input_cb_id = get_named_compile_time_arg_val("tilizer_input_cb_id");
    constexpr uint32_t tilizer_output_cb_id = get_named_compile_time_arg_val("tilizer_output_cb_id");
    constexpr uint32_t e_t_buffer_id = get_named_compile_time_arg_val("e_t_buffer_id");
    constexpr uint32_t expert_activation_cb_id = get_named_compile_time_arg_val("expert_activation_cb_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");

    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");
    constexpr uint32_t aligned_scores_page_size = get_named_compile_time_arg_val("aligned_scores_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");

    // Multicast coordinates for signaling tilizer cores
    constexpr uint32_t tilizer_mcast_start_x = get_named_compile_time_arg_val("tilizer_mcast_start_x");
    constexpr uint32_t tilizer_mcast_start_y = get_named_compile_time_arg_val("tilizer_mcast_start_y");
    constexpr uint32_t tilizer_mcast_end_x = get_named_compile_time_arg_val("tilizer_mcast_end_x");
    constexpr uint32_t tilizer_mcast_end_y = get_named_compile_time_arg_val("tilizer_mcast_end_y");
    constexpr uint32_t num_tilizer_cores = get_named_compile_time_arg_val("num_tilizer_cores");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;
    constexpr uint32_t tokens_per_device = tokens / dispatch_devices;

    // Aligned row size for expert_activation buffer (in bytes)
    constexpr uint32_t aligned_activation_row_bytes =
        ((2 * experts_per_device + 1) * sizeof(uint32_t) + l1_alignment - 1) / l1_alignment * l1_alignment;

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);     // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 1
    uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 3
    bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 4
    uint32_t tilizer_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);  // 5
    uint32_t tilizer_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);    // 6

    // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    const auto input_tensor_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_tensor_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto scores_tensor_addr_gen = TensorAccessor(scores_args, scores_tensor_address, indices_page_size);
    const auto mapping_tensor_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    const auto output_tensor_addr_gen =
        TensorAccessor(output_args, 0, output_page_size);  // output address not needed for reader

    // Read the mapping tensor page for this device (linearized_mesh_coord)
    // This gives us the expert -> device mapping from this device's perspective
    // Reserve all pages (tokens)

    cb_reserve_back(mapping_tensor_cb_id, mapping_pages);
    for (uint32_t i = 0; i < mapping_pages; i++) {
        noc_async_read_page(
            i, mapping_tensor_addr_gen, get_write_ptr(mapping_tensor_cb_id) + i * aligned_mapping_page_size);
    }

    cb_reserve_back(expert_activation_cb_id, tokens);
    init_expert_activation_buffer_async<selected_experts_k, tokens, experts_per_device, l1_alignment>(
        expert_activation_cb_id);

    noc_async_read_barrier();
    cb_push_back(mapping_tensor_cb_id, mapping_pages);
    cb_push_back(expert_activation_cb_id, tokens);

    // print_expert_activation_buffer<experts_per_device, l1_alignment>(expert_activation_cb_id, 0, tokens);

    // Get pointer to the mapping data
    uint16_t* expert_to_device_map = reinterpret_cast<uint16_t*>(
        get_read_ptr(mapping_tensor_cb_id) + linearized_mesh_coord * aligned_mapping_page_size);
    uint16_t local_expert_ids[experts_per_device];
    uint32_t local_expert_count = 0;
    for (uint32_t i = 0; i < experts; i++) {
        uint16_t expert_mesh_coord = expert_to_device_map[i];
        if (expert_mesh_coord == linearized_mesh_coord) {
            if (local_expert_count >= experts_per_device) {
                DPRINT << "Error: more than " << experts_per_device << " experts on device " << linearized_mesh_coord
                       << ENDL();
                ASSERT(false);
            }
            // DPRINT << "Device " << linearized_mesh_coord << " : Local expert " << local_expert_count << " is " << i
            // << ENDL();
            local_expert_ids[local_expert_count] = i;
            local_expert_count++;
        }
    }

    // indices is already in CB as it's sharded in L1
    uint32_t num_activated_tokens = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        // read in the token's source device's mapping
        uint32_t source_device =
            get_device_idx_from_global_token_idx<linearized_mesh_coord, tokens_per_device, mesh_rows, mesh_cols, axis>(
                t);

        // Pointer arithmetic: add byte offset first, then cast to element pointer
        uint16_t* source_device_mapping =
            reinterpret_cast<uint16_t*>(get_read_ptr(mapping_tensor_cb_id) + source_device * aligned_mapping_page_size);
        uint16_t* token_indices =
            reinterpret_cast<uint16_t*>(get_read_ptr(indices_tensor_cb_id) + t * aligned_indices_page_size);
        // scores tensor which is already in CB as it's sharded in L1 (bf16 = uint16_t)
        uint16_t* token_scores =
            reinterpret_cast<uint16_t*>(get_read_ptr(scores_tensor_cb_id) + t * aligned_scores_page_size);

        // Get pointer to this token's row in expert_activation buffer (using aligned row size)
        uint32_t expert_activation_l1_addr =
            get_write_ptr(expert_activation_cb_id) + num_activated_tokens * aligned_activation_row_bytes;
        volatile tt_l1_ptr uint32_t* expert_activation_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expert_activation_l1_addr);

        bool activated = false;
        for (uint32_t k = 0; k < selected_experts_k; k++) {
            // check if this is one of the local experts
            uint16_t selected_expert = token_indices[k];
            // if any of the selected experts :
            // 1) If this is a local expert, and the source device's mapping buffer believes this expert is on this
            // device, then we need to activate this token 2) If this is a local expert, but the source device's mapping
            // buffer believes this expert is not on this device, then we need to ignore this token 3) If this is not a
            // local expert, then we need to ignore this token
            for (uint32_t e = 0; e < local_expert_count; e++) {
                if (selected_expert == local_expert_ids[e]) {
                    if (source_device_mapping[selected_expert] == linearized_mesh_coord) {
                        // set the token id to t (only once per activated token)
                        if (!activated) {
                            expert_activation_l1_ptr[0] = t;
                        }
                        // set the expert index e to k
                        expert_activation_l1_ptr[1 + e] = k;
                        // set the score to the token's score (cast bf16 to uint32_t for storage)
                        expert_activation_l1_ptr[1 + experts_per_device + e] = static_cast<uint32_t>(token_scores[k]);
                        activated = true;
                    }
                }
            }
        }
        if (activated) {
            num_activated_tokens++;
        }
    }
    DPRINT << "Number of activated tokens: " << num_activated_tokens << ENDL();
    print_expert_activation_buffer<experts_per_device, l1_alignment>(expert_activation_cb_id, 0, tokens);

    // TODO: Implement selective tilize logic
    // 1. Read through all indices to determine which tokens belong to this device's experts
    // 2. For each chunk of tokens_per_chunk, read the matching tokens from sparse buffer
    // 3. Pack into tilizer input CB for compute kernel to tilize
}
