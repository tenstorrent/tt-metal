// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "moe_ring_common.h"

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
    Noc& noc, uint32_t src_local_l1_addr, uint64_t dst_noc_addr_multicast, uint32_t size, uint32_t num_dests) {
    // Device 2.0 migration: legacy primitive retained: dst_noc_addr_multicast is a precomposed uint64_t
    // NoC multicast address (returned by get_safe_multicast_noc_addr above). Noc::async_write_multicast
    // takes a typed Dst (MulticastEndpoint, TensorAccessor) plus separate dst_args; there is no wrapper
    // for a raw uint64_t multicast address today
    while (size > NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast(
            src_local_l1_addr, dst_noc_addr_multicast, NOC_MAX_BURST_SIZE, num_dests, true, noc.get_noc_id());
        src_local_l1_addr += NOC_MAX_BURST_SIZE;
        dst_noc_addr_multicast += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
    }
    noc_async_write_multicast(src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, false, noc.get_noc_id());
}

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT("cb_idx: {} tile_idx: {}\n", cb_idx, tile_idx);
    DPRINT("======\n");
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT(
            "{} : {}\n",
            r,
            TileSlice(
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
                untilize));
    }
    DPRINT("++++++\n");
}

template <
    uint32_t LinearizedMeshCoord,
    uint32_t TokensPerDevice,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup Axis>
inline uint32_t get_device_idx_from_global_token_idx(const uint32_t t) {
    [[maybe_unused]] constexpr uint32_t Replicate_Group = (Axis == ReplicateGroup::NONE)   ? MeshRows * MeshCols
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
    [[maybe_unused]] constexpr uint32_t shared_cb_num_pages = get_named_compile_time_arg_val("shared_cb_num_pages");
    // One chunk slot of the staging CB per chunk (see tilize_compute.cpp)
    constexpr uint32_t chunk_slot_pages = get_named_compile_time_arg_val("chunk_slot_pages");

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
    constexpr uint32_t experts_per_device = get_named_compile_time_arg_val("experts_per_device");

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

    // Chunk halves of the ring cores' input buffer (2 today, R + 1 with R prefill rings) and the drain's credit
    // semaphore per half: a ring core's dm1 increments half g % chunk_halves after consuming chunk g.
    constexpr uint32_t chunk_halves = get_named_compile_time_arg_val("chunk_halves");
    static_assert(chunk_halves >= 2 && chunk_halves <= moe_ring::rings::MAX_CHUNK_HALVES, "chunk_halves out of range");
    constexpr uint32_t half_free_semaphore_ids[moe_ring::rings::MAX_CHUNK_HALVES] = {
        get_named_compile_time_arg_val("half_free_semaphore_id_0"),
        get_named_compile_time_arg_val("half_free_semaphore_id_1"),
        get_named_compile_time_arg_val("half_free_semaphore_id_2"),
        get_named_compile_time_arg_val("half_free_semaphore_id_3"),
        get_named_compile_time_arg_val("half_free_semaphore_id_4")};

    constexpr uint32_t matmul_mcast_start_x = get_named_compile_time_arg_val("matmul_mcast_start_x");
    constexpr uint32_t matmul_mcast_start_y = get_named_compile_time_arg_val("matmul_mcast_start_y");
    constexpr uint32_t matmul_mcast_end_x = get_named_compile_time_arg_val("matmul_mcast_end_x");
    constexpr uint32_t matmul_mcast_end_y = get_named_compile_time_arg_val("matmul_mcast_end_y");
    constexpr uint32_t matmul_bounding_box_num_cores = get_named_compile_time_arg_val("matmul_bounding_box_num_cores");

    // Semaphores
    constexpr uint32_t tilize_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("tilize_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    constexpr uint32_t initial_gather_semaphore_id = get_named_compile_time_arg_val("initial_gather_semaphore_id");
    // The secondary multicaster (the first core of the second gather group): it multicasts its group's columns of
    // every chunk on NoC0 while the drain multicasts the first group's on NoC1.
    constexpr uint32_t secondary_mcaster_noc_x = get_named_compile_time_arg_val("secondary_mcaster_noc_x");
    constexpr uint32_t secondary_mcaster_noc_y = get_named_compile_time_arg_val("secondary_mcaster_noc_y");
    constexpr bool two_mcasters = num_tilize_cores > 1;
    // The steady-state gather protocol (every chunk past the first): the drain counts the other cores' sub-chunks per
    // staging slot, and tells them how many chunks it has multicast, so a core gathers chunk m as soon as the drain's
    // slot m % chunk_halves is free (popped before the multicast of chunk m + 1 - chunk_halves) -- under the drain's
    // multicast of the previous chunk, not after its go.
    constexpr uint32_t feed_go_semaphore_id = get_named_compile_time_arg_val("feed_go_semaphore_id");
    constexpr uint32_t gather_semaphore_ids[moe_ring::rings::MAX_CHUNK_HALVES] = {
        get_named_compile_time_arg_val("gather_semaphore_id_0"),
        get_named_compile_time_arg_val("gather_semaphore_id_1"),
        get_named_compile_time_arg_val("gather_semaphore_id_2"),
        get_named_compile_time_arg_val("gather_semaphore_id_3"),
        get_named_compile_time_arg_val("gather_semaphore_id_4")};

    // When local_output=1 (moe_compute LocalOutput) no combine reads the routing: BRISC keeps its (token, expert,
    // k slot) hits as a pair list (moe_ring::token_list) for the drain's packed lists and builds no activation rows.
    constexpr bool local_output = get_named_compile_time_arg_val("local_output") == 1;

    Semaphore<> tilize_chunk_ready_sem(tilize_chunk_ready_semaphore_id);
    Semaphore<> matmul_chunk_ready_sem(matmul_chunk_ready_semaphore_id);
    // On the secondary mcaster: the drain's count of the input halves it has seen credited (one per chunk past the
    // first chunk_halves), so the secondary multicasts into a half only after the ring cores consumed its previous
    // chunk too. (The semaphore id is the historical "initial gather" one.)
    Semaphore<> half_ok_sem(initial_gather_semaphore_id);
    Semaphore<> feed_go_sem(feed_go_semaphore_id);

    // Device 2.0 migration: legacy primitives retained: these raw L1 semaphore addresses are
    // used as bases for multicast destinations (set_multicast / get_safe_multicast_noc_addr /
    // get_noc_addr) and for direct noc_semaphore_set with the legacy address-taking overload.
    uint32_t tilize_chunk_ready_semaphore_addr = get_semaphore(tilize_chunk_ready_semaphore_id);

    // Noc typed wrappers
    Noc noc_obj(noc_index);
    Noc noc_alt_obj(1 - noc_index);

    // CircularBuffer typed wrappers
    CircularBuffer cb_tilize_output(tilize_output_cb_id);
    CircularBuffer cb_per_expert_total_tokens(per_expert_total_tokens_cb_id);
    CircularBuffer cb_indices_tensor(indices_tensor_cb_id);
    CircularBuffer cb_scores_tensor(scores_tensor_cb_id);
    CircularBuffer cb_mapping_tensor(mapping_tensor_cb_id);
    CircularBuffer cb_brisc_e_t(brisc_e_t_cb_id);
    CircularBuffer cb_brisc_expert_counts(brisc_expert_counts_cb_id);
    CircularBuffer cb_brisc_expert_activation(brisc_expert_activation_cb_id);
    CircularBuffer cb_brisc_activated_count(brisc_activated_count_cb_id);

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    [[maybe_unused]] uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 0 - not used by writer
    [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 1 - not used by writer
    [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 2 - not used by writer
    [[maybe_unused]] uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 3 - not used by writer
    [[maybe_unused]] uint32_t per_expert_total_tokens_output_tensor_address =
        get_arg_val<uint32_t>(rt_args_idx++);  // 4 not used by writer
    [[maybe_unused]] uint32_t expert_activation_output_address =
        get_arg_val<uint32_t>(rt_args_idx++);                                             // 5 - not used by writer
    [[maybe_unused]] uint32_t e_t_output_address = get_arg_val<uint32_t>(rt_args_idx++);  // 6 - not used by writer
    bool is_drain_tilize_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);               // 7
    bool is_secondary_mcaster = (bool)get_arg_val<uint32_t>(rt_args_idx++);               // 8
    uint32_t initial_mcast_gather_core_nox_x = get_arg_val<uint32_t>(rt_args_idx++);      // 9
    uint32_t initial_mcast_gather_core_nox_y = get_arg_val<uint32_t>(rt_args_idx++);      // 10
    uint32_t global_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);               // 11
    uint32_t mcast_group_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);          // 12
    uint32_t mcast_group_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);            // 13
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                        // 14
    uint32_t core_token_start = get_arg_val<uint32_t>(rt_args_idx++);                     // 15
    uint32_t core_token_end = get_arg_val<uint32_t>(rt_args_idx++);                       // 16
    [[maybe_unused]] uint32_t tilize_core_idx = get_arg_val<uint32_t>(rt_args_idx++);     // 17 - not used by writer

    // Constants
    constexpr uint32_t one_page = 1;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t element_size = tilize_output_page_size / (TILE_HEIGHT * TILE_WIDTH);
    constexpr uint32_t tile_width_bytes = TILE_WIDTH * element_size;

    // For parallel metadata processing - BRISC processes second half of this core's token range
    // Note: These are computed at runtime based on core_token_start/end in Step 3
    [[maybe_unused]] constexpr uint32_t brisc_token_start = tokens / 2;
    [[maybe_unused]] constexpr uint32_t brisc_token_end = tokens;
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
    // ceil(tokens_this_core/2): BRISC's token count; must match tilize_reader.
    uint32_t brisc_tokens_capacity = tokens_this_core - tokens_this_core / 2;

    // Wait for NCRISC to finish reading the mapping tensor
    cb_mapping_tensor.wait_front(num_devices);

    // Get mapping base pointer (read by NCRISC)
    const uint32_t mapping_base = cb_mapping_tensor.get_read_ptr();

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

    // Reserve BRISC's e_t buffer (single page contains all experts' token lists; under local_output its pair list)
    cb_brisc_e_t.reserve_back(one_page);
    const uint32_t brisc_e_t_buffer_base = cb_brisc_e_t.get_write_ptr();
    [[maybe_unused]] uint32_t brisc_num_pairs = 0;

    [[maybe_unused]] uint32_t brisc_expert_activation_base = 0;
    if constexpr (!local_output) {
        // Reserve BRISC's expert_activation buffer (single page contains all activation rows)
        cb_brisc_expert_activation.reserve_back(one_page);
        brisc_expert_activation_base = cb_brisc_expert_activation.get_write_ptr();

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
    }

    // Indices and scores accessible via CB (drain has shard, non-drain read via NOC in reader)
    const uint32_t indices_base = cb_indices_tensor.get_read_ptr();
    const uint32_t scores_base = cb_scores_tensor.get_read_ptr();

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
        [[maybe_unused]] uint32_t* brisc_activation_l1_ptr = nullptr;
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
                    if constexpr (local_output) {
                        // Pair list in arrival order (the drain scatters it into expert e's packed segment).
                        uint32_t* pair = reinterpret_cast<uint32_t*>(
                            brisc_e_t_buffer_base + brisc_num_pairs * moe_ring::token_list::PAIR_BYTES);
                        pair[0] = t;
                        pair[1] = (e << moe_ring::token_list::PAIR_EXPERT_SHIFT) | k;
                        brisc_num_pairs++;
                    } else {
                        // First activation for this token - set up pointer and write token id
                        if (!activated) {
                            brisc_activation_l1_ptr = reinterpret_cast<uint32_t*>(
                                brisc_expert_activation_base +
                                brisc_num_activated_tokens * aligned_activation_row_bytes);
                            brisc_activation_l1_ptr[0] = t;
                        }

                        // Write k-index and score for this expert
                        brisc_activation_l1_ptr[1 + e] = k;
                        brisc_activation_l1_ptr[1 + experts_per_device + e] = static_cast<uint32_t>(token_scores[k]);

                        // Write to BRISC's e_t buffer (16B aligned entries): word 0 token id, word 1 the
                        // token's k slot for this expert (same entry format as the NCRISC buffer; the
                        // merge copies whole entries).
                        const uint32_t brisc_e_t_offset =
                            (e * brisc_tokens_capacity + brisc_num_tokens_per_expert[e]) * e_t_entry_size;
                        uint32_t* brisc_e_t_entry =
                            reinterpret_cast<uint32_t*>(brisc_e_t_buffer_base + brisc_e_t_offset);
                        brisc_e_t_entry[0] = t;
                        brisc_e_t_entry[1] = k;
                    }
                    activated = true;
                    brisc_num_tokens_per_expert[e]++;
                    break;
                }
            }
        }

        if (activated) {
            brisc_num_activated_tokens++;
        }
    }

    // Push BRISC's e_t buffer (no -1 cap needed, NCRISC will cap final merged buffer) or its pair list
    cb_brisc_e_t.push_back(one_page);

    if constexpr (!local_output) {
        // Push BRISC's expert_activation buffer
        cb_brisc_expert_activation.push_back(one_page);
    }

    // Push BRISC's per-expert counts to CB for NCRISC to read
    cb_brisc_expert_counts.reserve_back(one_page);
    uint32_t* brisc_counts_ptr = reinterpret_cast<uint32_t*>(cb_brisc_expert_counts.get_write_ptr());
    for (uint32_t e = 0; e < experts_per_device; e++) {
        brisc_counts_ptr[e] = brisc_num_tokens_per_expert[e];
    }
    cb_brisc_expert_counts.push_back(one_page);

    if constexpr (!local_output) {
        // Push BRISC's activated token count
        cb_brisc_activated_count.reserve_back(one_page);
        *reinterpret_cast<uint32_t*>(cb_brisc_activated_count.get_write_ptr()) = brisc_num_activated_tokens;
        cb_brisc_activated_count.push_back(one_page);
    }

    // Wait for reader to push per-expert token counts (includes merged NCRISC + BRISC counts)
    cb_per_expert_total_tokens.wait_front(1);
    volatile tt_l1_ptr uint32_t* per_expert_counts =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_per_expert_total_tokens.get_read_ptr());

    // Read per-expert token counts into local array
    uint32_t num_tokens_per_expert[experts_per_device];
    for (uint32_t e = 0; e < experts_per_device; e++) {
        num_tokens_per_expert[e] = per_expert_counts[e];
    }

    /************************************************************************/
    /* Synchronization setup for signalling between tilize and matmul cores */
    /************************************************************************/

    // Chunk g lands in half g % chunk_halves of the ring cores' input buffer. Before the drain multicasts into a half
    // it waits for every ring core's credit for the chunk that last used it (one semaphore per half, num_matmul_cores
    // credits per chunk). The other tilize cores do not wait for it: their sub-chunks land in the drain's staging.
    volatile tt_l1_ptr uint32_t* half_free_sem_ptr[moe_ring::rings::MAX_CHUNK_HALVES];
    uint32_t half_chunks_landed[moe_ring::rings::MAX_CHUNK_HALVES];
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        half_free_sem_ptr[h] =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(half_free_semaphore_ids[h]));
        half_chunks_landed[h] = 0;
    }

    // Semaphore we use to signal to matmul cores that a chunk has arrived
    uint32_t matmul_chunk_ready_semaphore_set_value = 1;

    // How many chunks we've sent to matmul so far
    uint32_t num_chunks_sent = 0;

    uint32_t matmul_chunk_input_cb_base_addr = cb_tilize_output.get_read_ptr();
    uint32_t first_half_buffer_addr = matmul_chunk_input_cb_base_addr;
    const uint32_t chunk_half_bytes = tiles_per_global_chunk * tilize_output_page_size;

    // For synchronization between the drain-sync core and non-drain-sync cores
    // The secondary mcaster tells the drain, once per chunk, that its half of the chunk has landed (== 7 ==)
    uint64_t tilize_chunk_ready_drain_semaphore_noc_addr =
        get_noc_addr(drain_core_noc_x, drain_core_noc_y, tilize_chunk_ready_semaphore_addr, noc_index);
    // The drain tells the secondary, once per chunk past the first chunk_halves, that the chunk's input half is
    // credited (== 1 ==)
    const uint64_t secondary_half_ok_noc_addr = get_noc_addr(
        secondary_mcaster_noc_x, secondary_mcaster_noc_y, get_semaphore(initial_gather_semaphore_id), noc_index);
    // The gathers travel on the other NoC (== 3 ==) into this core's group multicaster: its per-slot gather counts
    // addressed for that NoC (the senders), and a multicaster's own view of them
    const bool secondary_role = two_mcasters && is_secondary_mcaster;
    const uint32_t own_group_others = !two_mcasters          ? 0u
                                      : is_drain_tilize_core ? primary_mcast_gather_group_num_cores - 1
                                                             : secondary_mcast_gather_group_num_cores - 1;
    uint64_t gather_sem_alt_noc_addr[moe_ring::rings::MAX_CHUNK_HALVES];
    volatile tt_l1_ptr uint32_t* gather_sem_ptr[moe_ring::rings::MAX_CHUNK_HALVES];
    uint32_t slot_gathers[moe_ring::rings::MAX_CHUNK_HALVES];  // multicaster: chunks gathered per slot so far
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        gather_sem_alt_noc_addr[h] = get_noc_addr(
            initial_mcast_gather_core_nox_x,
            initial_mcast_gather_core_nox_y,
            get_semaphore(gather_semaphore_ids[h]),
            1 - noc_index);
        gather_sem_ptr[h] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_semaphore_ids[h]));
        slot_gathers[h] = 0;
    }

    // This multicaster's NoC (the drain NoC1, the secondary NoC0), the bytes it sends per chunk (its group's
    // columns; the whole chunk when it is the only tilize core) and the destination in each chunk half of the ring
    // cores' input buffer (its group's columns start at this core's global tile offset)
    Noc& mcast_noc_obj = secondary_role ? noc_alt_obj : noc_obj;
    const uint8_t mcast_noc = secondary_role ? (1 - noc_index) : noc_index;
    const uint32_t bytes_to_mcast =
        (two_mcasters ? tiles_per_mcast_group_chunk : tiles_per_global_chunk) * tilize_output_page_size;
    uint64_t mcast_dest_addr[moe_ring::rings::MAX_CHUNK_HALVES];
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        mcast_dest_addr[h] = get_safe_multicast_noc_addr(
            matmul_mcast_start_x,
            matmul_mcast_start_y,
            matmul_mcast_end_x,
            matmul_mcast_end_y,
            first_half_buffer_addr + h * chunk_half_bytes + global_tile_offset * tilize_output_page_size,
            mcast_noc);
    }

    /* start loop iterations */

    // Process each expert's chunks
    // Order matches reader: all chunks for expert 0, then expert 1, etc.
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t num_expert_tokens = num_tokens_per_expert[e];
        uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++) {
            // Study zones (MOE_ZONES): this chunk's phases, recorded inside the profiler window only
            const bool zone_on = moe_ring::zones::in_window(num_chunks_sent);
            {
                // Wait for compute to push this chunk's slot (its tiles_per_local_chunk tiles at the slot's start)
                MOE_ZONE_IF(zone_on, "mz_w_tilized");
                cb_tilize_output.wait_front(chunk_slot_pages);
            }
            uint32_t l1_read_addr = cb_tilize_output.get_read_ptr();

            /*
             * Send chunks to MM cores (chunk g lands in half g % chunk_halves of the ring cores' input buffer):
             * 1) the drain waits for the half to be credited by every ring core (skipped for the first chunk_halves
             *    chunks: every half starts empty) and tells the secondary multicaster so (half_ok)
             * 2) with several tilize cores two multicasters send every chunk: the drain its gather group's columns
             *    over NoC1, the secondary multicaster its group's over NoC0; one tilize core sends the whole chunk
             * 3) the other cores wait until their multicaster has multicast chunk m + 1 - chunk_halves (feed_go), so
             *    its staging slot m % chunk_halves is free, then send their sub-chunk into it over NoC0
             * 4) ... and increment the multicaster's gather count of that slot (NoC0, behind the data)
             * 5) a multicaster waits until its slot's gather count says every sub-chunk of its group landed
             * 6) a multicaster mcasts its group's tiles into the ring cores' half (linked, its NoC)
             * 7) the secondary barriers its NoC0 multicast and increments the drain's count of second halves landed
             * 8) the drain waits for that count
             * 9) the drain mcasts to the MM cores that the chunk has arrived
             * 10) the drain multicasts feed_go = the number of chunks it has multicast
             *
             * NOTE: a linked multicast needs its NoC idle on its core: the drain's NoC1 carries its multicast and its
             *       semaphore multicasts only (the drain's reader reads rows on NoC0), the secondary's NoC0 carries its
             *       multicast only (its reader reads rows on NoC1, the gathers into it are the senders' NoC0 writes).
             */

            // == 1 ==
            // skip for the first chunk_halves chunks (every half is initially empty)
            const uint32_t chunk_half = num_chunks_sent % chunk_halves;
            // Only the drain waits: the credit gates its multicast into the ring cores' half; the other tilize cores'
            // gather into the drain's staging slot is gated by the slot rule (== 3b ==) and their own staging CB.
            if (num_chunks_sent >= chunk_halves && is_drain_tilize_core) {
                // every ring core has consumed the chunk that last used this half (credits are cumulative per half)
                {
                    MOE_ZONE_IF(zone_on, "mz_w_half");
                    if constexpr (!MOE_STUDY_FAULT(X_DRAIN_NO_HALF_WAIT)) {
                        noc_semaphore_wait_min(
                            half_free_sem_ptr[chunk_half], num_matmul_cores * half_chunks_landed[chunk_half]);
                    }
                }
                if constexpr (two_mcasters) {
                    // the secondary multicaster writes into this half too: pass the credit on (one count per chunk)
                    noc_semaphore_inc(secondary_half_ok_noc_addr, 1, noc_index);
                }
            }

            // == 2 ==
            if (is_drain_tilize_core || secondary_role) {
                // == 5 ==
                // wait until the group's other cores' sub-chunks landed in this chunk's slot (per-slot counts)
                {
                    MOE_ZONE_IF(zone_on, "mz_w_gather");
                    slot_gathers[chunk_half] += 1;
                    noc_semaphore_wait_min(gather_sem_ptr[chunk_half], own_group_others * slot_gathers[chunk_half]);
                }
                if (secondary_role && num_chunks_sent >= chunk_halves) {
                    // the drain has seen this chunk's input half credited (== 1 ==)
                    MOE_ZONE_IF(zone_on, "mz_w_half_ok");
                    half_ok_sem.wait_min(num_chunks_sent + 1 - chunk_halves);
                }

                MOE_STUDY_DELAY(W_BEFORE_MCAST);
                // == 6 ==
                // mcast this group's tiles of the chunk into the ring cores' half (linked, this multicaster's NoC)
                {
                    MOE_ZONE_IF(zone_on, "mz_w_mcast");
                    noc_async_write_linked_multicast(
                        mcast_noc_obj,
                        l1_read_addr,
                        mcast_dest_addr[chunk_half],
                        bytes_to_mcast,
                        matmul_bounding_box_num_cores);
                }
                if constexpr (two_mcasters) {
                    if (secondary_role) {
                        // == 7 ==
                        // the drain signals the MM cores on NoC1: this multicast (NoC0) must have landed first
                        noc_alt_obj.async_write_barrier();
                        MOE_STUDY_DELAY(W_SECONDARY_BEFORE_INC);
                        // Device 2.0 migration: legacy primitive retained: the drain's count is a precomposed
                        // uint64_t NoC address (tilize_chunk_ready_drain_semaphore_noc_addr)
                        noc_semaphore_inc(tilize_chunk_ready_drain_semaphore_noc_addr, 1, noc_index);
                    } else {
                        // == 8 ==
                        // wait for the secondary's half of this chunk (cumulative count)
                        MOE_ZONE_IF(zone_on, "mz_w_second_half");
                        tilize_chunk_ready_sem.wait_min(num_chunks_sent + 1);
                    }
                }
            } else {
                // == 3 ==
                // send to proper offset on this core's group multicaster, over the OTHER NoC: the multicaster's linked
                // multicast of the previous chunk may still run, and the next chunk's sub-chunk landing under it is
                // what overlaps the gather with it
                uint32_t gather_addr = l1_read_addr + mcast_group_tile_offset * tilize_output_page_size;
                // the multicaster's slot for this chunk was last used by chunk num_chunks_sent - chunk_halves, popped
                // before the drain multicast the chunk after it (the drain's count follows the secondary's, == 8 ==)
                if (num_chunks_sent + 2 > chunk_halves) {
                    MOE_ZONE_IF(zone_on, "mz_w_go_wait");
                    if constexpr (!MOE_STUDY_FAULT(X_GATHER_NO_GO_WAIT)) {
                        feed_go_sem.wait_min(num_chunks_sent + 2 - chunk_halves);
                    }
                }
                MOE_STUDY_DELAY(W_GATHER_BEFORE_WRITE);
                noc_alt_obj.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    UnicastEndpoint{},
                    tiles_per_local_chunk * tilize_output_page_size,
                    {},
                    {.noc_x = initial_mcast_gather_core_nox_x,
                     .noc_y = initial_mcast_gather_core_nox_y,
                     .addr = gather_addr});
                noc_alt_obj.async_write_barrier();

                MOE_STUDY_DELAY(W_GATHER_BEFORE_INC);
                // == 4 ==
                // increment the multicaster's gather count of this slot (same NoC as the data)
                // Device 2.0 migration: legacy primitive retained: the per-slot gather count is a precomposed
                // uint64_t NoC address (gather_sem_alt_noc_addr)
                noc_semaphore_inc(gather_sem_alt_noc_addr[chunk_half], 1, 1 - noc_index);
            }

            {
                MOE_ZONE_IF(zone_on, "mz_w_signal");
                if (is_drain_tilize_core) {
                    MOE_STUDY_DELAY(W_BEFORE_SIGNAL);
                    // == 9 ==
                    // signal to MM cores that entire chunk has arrived

                    // set local value
                    matmul_chunk_ready_sem.set(matmul_chunk_ready_semaphore_set_value);
                    matmul_chunk_ready_semaphore_set_value++;

                    // mcast sem set
                    set_multicast_safe(
                        matmul_chunk_ready_sem,
                        noc_obj,
                        matmul_mcast_start_x,
                        matmul_mcast_start_y,
                        matmul_mcast_end_x,
                        matmul_mcast_end_y,
                        matmul_bounding_box_num_cores);

                    // == 10 ==
                    if constexpr (two_mcasters) {
                        MOE_STUDY_DELAY(W_BEFORE_GO);
                        // the count of chunks multicast (this one included): the gathering cores send chunk m into
                        // their multicaster once it reaches m + 2 - chunk_halves (the slot m % chunk_halves is free)
                        feed_go_sem.set(num_chunks_sent + 1);
                        set_multicast_safe(
                            feed_go_sem,
                            noc_obj,
                            tilize_mcast_start_x,
                            tilize_mcast_start_y,
                            tilize_mcast_end_x,
                            tilize_mcast_end_y,
                            tilize_bounding_box_num_cores - 1);
                    }
                }
            }

            {
                // we already barrier when using (1 - noc_index), so just need to flush on noc_index here
                MOE_ZONE_IF(zone_on, "mz_w_flush");
                if constexpr (!MOE_STUDY_FAULT(X_DRAIN_NO_FLUSH)) {
                    noc_obj.async_writes_flushed();
                }
            }

            MOE_STUDY_DELAY(W_BEFORE_POP);
            // pop this chunk's slot (the next chunk may already be tilized into the next slot)
            cb_tilize_output.pop_front(chunk_slot_pages);
            half_chunks_landed[chunk_half] += 1;
            num_chunks_sent++;
        }
    }

    if (is_drain_tilize_core) {
        // Trailing waits: every ring core's credit has landed before this kernel exits, so no increment of this
        // launch can reach the semaphores of the next one.
        for (uint32_t h = 0; h < chunk_halves; ++h) {
            noc_semaphore_wait_min(half_free_sem_ptr[h], num_matmul_cores * half_chunks_landed[h]);
        }
    }

    // Pop the per-expert counts (cleanup).
    cb_per_expert_total_tokens.pop_front(one_page);

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
    // the gathers and their semaphore increments went over the other NoC (non-posted: their responses must be back)
    noc_alt_obj.async_write_barrier();
    noc_alt_obj.async_atomic_barrier();
}
