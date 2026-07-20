// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "api/tensor/noc_traits.h"

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
FORCE_INLINE void init_expert_activation_buffer_async(Noc& noc, CircularBuffer& cb) {
    constexpr uint32_t row_elements = 2 * experts_per_device + 1;
    constexpr uint32_t row_size_bytes_unaligned = row_elements * sizeof(uint32_t);
    // Align row size to l1_alignment for NOC transfer efficiency
    constexpr uint32_t aligned_row_size_bytes =
        ((row_size_bytes_unaligned + l1_alignment - 1) / l1_alignment) * l1_alignment;

    // Minimum transfer size for maximum NOC throughput (~27.8 bytes/cycle)
    constexpr uint32_t MIN_EFFICIENT_BYTES = 512;

    uint32_t l1_write_addr = cb.get_write_ptr();
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

    // Device 2.0 migration: legacy primitive retained: get_noc_addr(local_l1_addr) is used to
    // form a self-loopback NoC source for L1->L1 broadcast-doubling copies
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
        // Device 2.0 migration: legacy primitive retained: src is a precomposed uint64_t
        // self-loopback NoC address (L1->L1 broadcast-doubling)
        noc_async_read(src_noc_addr, dest_addr, bytes_to_copy);
        noc.async_read_barrier();  // Must barrier before next doubling iteration

        rows_filled += rows_to_copy;
    }

    // Phase 2: Parallel dispatch remaining copies at max efficiency (no barriers between)
    if (rows_filled < tokens) {
        uint32_t chunk_rows = rows_filled;  // Copy this many rows at a time
        uint32_t chunk_bytes = chunk_rows * aligned_row_size_bytes;

        // Dispatch all full chunks in parallel
        while (rows_filled + chunk_rows <= tokens) {
            uint32_t dest_addr = l1_write_addr + rows_filled * aligned_row_size_bytes;
            // Device 2.0 migration: legacy primitive retained: src is a precomposed uint64_t
            // self-loopback NoC address (L1->L1 broadcast-doubling)
            noc_async_read(src_noc_addr, dest_addr, chunk_bytes);
            rows_filled += chunk_rows;
        }

        // Handle remainder rows (if tokens not divisible by chunk_rows)
        if (rows_filled < tokens) {
            uint32_t remainder_rows = tokens - rows_filled;
            uint32_t remainder_bytes = remainder_rows * aligned_row_size_bytes;
            uint32_t dest_addr = l1_write_addr + rows_filled * aligned_row_size_bytes;
            // Device 2.0 migration: legacy primitive retained: src is a precomposed uint64_t
            // self-loopback NoC address (L1->L1 broadcast-doubling)
            noc_async_read(src_noc_addr, dest_addr, remainder_bytes);
        }
    }

    // Note: Caller must call noc.async_read_barrier() before using the buffer,
    // then cb.push_back(tokens) when ready to make data available.
}

// Debug print function for expert_activation buffer
// Prints each row showing: [token_id | expert_activations... | scores...]
// start_token/end_token allow printing a subset of rows
template <uint32_t experts_per_device, uint32_t l1_alignment>
FORCE_INLINE void print_expert_activation_buffer(
    CircularBuffer& cb, uint32_t start_token = 0, uint32_t end_token = 0xFFFFFFFF) {
    constexpr uint32_t row_elements = 2 * experts_per_device + 1;
    constexpr uint32_t row_size_bytes_unaligned = row_elements * sizeof(uint32_t);
    constexpr uint32_t aligned_row_size_bytes =
        ((row_size_bytes_unaligned + l1_alignment - 1) / l1_alignment) * l1_alignment;
    constexpr uint32_t aligned_row_elements = aligned_row_size_bytes / sizeof(uint32_t);

    volatile tt_l1_ptr uint32_t* buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_read_ptr());

    DPRINT("=== Expert Activation Buffer ===\n");
    DPRINT(
        "Row format: [token_id | act_0..act_{} | score_0...score_{}]\n",
        experts_per_device - 1,
        experts_per_device - 1);

    for (uint32_t t = start_token; t < end_token; t++) {
        uint32_t base = t * aligned_row_elements;

        // Token ID (stored as uint32_t, but -1 means unset)
        uint32_t token_id = buffer[base];
        DPRINT("T{}: [", t);
        if (token_id == static_cast<uint32_t>(-1)) {
            DPRINT("-1");
        } else {
            DPRINT("{}", token_id);
        }
        DPRINT(" |");

        // Expert activations (k+1 means not activated, 0..k-1 means activated with that k-index)
        for (uint32_t e = 0; e < experts_per_device; e++) {
            DPRINT(" {}", buffer[base + 1 + e]);
        }
        DPRINT(" |");

        // Scores
        for (uint32_t e = 0; e < experts_per_device; e++) {
            DPRINT(" {}", bf16_t(static_cast<uint16_t>(buffer[base + 1 + experts_per_device + e])));
        }
        DPRINT("]\n");
    }
    DPRINT("================================\n");
}

// Print the E-T buffer (Expert-Token buffer)
// Format: E sections, each with up to T token IDs (16B aligned entries), terminated by -1
template <uint32_t experts_per_device, uint32_t tokens, uint32_t entry_size>
void print_e_t_buffer(CircularBuffer& cb) {
    uint32_t buffer_base = cb.get_read_ptr();

    DPRINT("=== E-T Buffer (Expert -> Tokens) ===\n");
    for (uint32_t e = 0; e < experts_per_device; e++) {
        DPRINT("Expert {}: [", e);
        uint32_t expert_base = buffer_base + e * tokens * entry_size;
        bool first = true;
        for (uint32_t i = 0; i < tokens; i++) {
            uint32_t token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expert_base + i * entry_size);
            if (token_id == static_cast<uint32_t>(-1)) {
                DPRINT(" -1]\n");
                break;
            }
            if (!first) {
                DPRINT(", ");
            }
            DPRINT("{}", token_id);
            first = false;
        }
    }
    DPRINT("======================================\n");
}

void kernel_main() {
    // Compile-time arguments

    // CBs
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t tilize_input_cb_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t e_t_cb_id = get_named_compile_time_arg_val("e_t_cb_id");
    constexpr uint32_t expert_activation_cb_id = get_named_compile_time_arg_val("expert_activation_cb_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t total_chunks_cb_id = get_named_compile_time_arg_val("total_chunks_cb_id");
    constexpr uint32_t brisc_e_t_cb_id = get_named_compile_time_arg_val("brisc_e_t_cb_id");
    constexpr uint32_t brisc_expert_counts_cb_id = get_named_compile_time_arg_val("brisc_expert_counts_cb_id");
    constexpr uint32_t brisc_expert_activation_cb_id = get_named_compile_time_arg_val("brisc_expert_activation_cb_id");
    constexpr uint32_t brisc_activated_count_cb_id = get_named_compile_time_arg_val("brisc_activated_count_cb_id");
    constexpr uint32_t remote_counts_cb_id = get_named_compile_time_arg_val("remote_counts_cb_id");

    // Alignment
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t e_t_entry_size = get_named_compile_time_arg_val("e_t_entry_size");

    // Number of pages
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    // Page sizes
    [[maybe_unused]] constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t per_expert_total_tokens_output_page_size =
        get_named_compile_time_arg_val("per_expert_total_tokens_output_page_size");
    constexpr uint32_t expert_activation_output_page_size =
        get_named_compile_time_arg_val("expert_activation_output_page_size");
    constexpr uint32_t e_t_output_page_size = get_named_compile_time_arg_val("e_t_output_page_size");

    // Aligned page sizes
    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_scores_page_size = get_named_compile_time_arg_val("aligned_scores_page_size");

    // General info
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t remote_counts_entry_size = get_named_compile_time_arg_val("remote_counts_entry_size");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t experts_per_device = get_named_compile_time_arg_val("experts_per_device");

    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");

    // Chunk info
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    // Mesh
    [[maybe_unused]] constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    // Multicast coordinates for drain tilize to non-drain tilize synchronization
    constexpr uint32_t drain_core_noc_x = get_named_compile_time_arg_val("drain_core_noc_x");
    constexpr uint32_t drain_core_noc_y = get_named_compile_time_arg_val("drain_core_noc_y");

    // T multicast coordinates
    constexpr uint32_t num_tilize_cores = get_named_compile_time_arg_val("num_tilize_cores");

    constexpr uint32_t tilize_mcast_start_x = get_named_compile_time_arg_val("tilize_mcast_start_x");
    constexpr uint32_t tilize_mcast_start_y = get_named_compile_time_arg_val("tilize_mcast_start_y");
    constexpr uint32_t tilize_mcast_end_x = get_named_compile_time_arg_val("tilize_mcast_end_x");
    constexpr uint32_t tilize_mcast_end_y = get_named_compile_time_arg_val("tilize_mcast_end_y");
    constexpr uint32_t tilize_bounding_box_num_cores = get_named_compile_time_arg_val("tilize_bounding_box_num_cores");

    // Multicast coordinates for signalling MM cores
    [[maybe_unused]] constexpr uint32_t num_matmul_cores = get_named_compile_time_arg_val("num_matmul_cores");

    constexpr uint32_t matmul_mcast_start_x = get_named_compile_time_arg_val("matmul_mcast_start_x");
    constexpr uint32_t matmul_mcast_start_y = get_named_compile_time_arg_val("matmul_mcast_start_y");
    constexpr uint32_t matmul_mcast_end_x = get_named_compile_time_arg_val("matmul_mcast_end_x");
    constexpr uint32_t matmul_mcast_end_y = get_named_compile_time_arg_val("matmul_mcast_end_y");
    constexpr uint32_t matmul_bounding_box_num_cores = get_named_compile_time_arg_val("matmul_bounding_box_num_cores");

    // All worker cores multicast coordinates
    constexpr uint32_t all_worker_cores_mcast_start_x =
        get_named_compile_time_arg_val("all_worker_cores_mcast_start_x");
    constexpr uint32_t all_worker_cores_mcast_start_y =
        get_named_compile_time_arg_val("all_worker_cores_mcast_start_y");
    constexpr uint32_t all_worker_cores_mcast_end_x = get_named_compile_time_arg_val("all_worker_cores_mcast_end_x");
    constexpr uint32_t all_worker_cores_mcast_end_y = get_named_compile_time_arg_val("all_worker_cores_mcast_end_y");
    constexpr uint32_t all_worker_cores_bounding_box_num_cores =
        get_named_compile_time_arg_val("all_worker_cores_bounding_box_num_cores");

    // Coordinates for combine signalling seminc
    constexpr uint32_t combine_sync_noc_x = get_named_compile_time_arg_val("combine_sync_noc_x");
    constexpr uint32_t combine_sync_noc_y = get_named_compile_time_arg_val("combine_sync_noc_y");

    // Semaphores
    constexpr uint32_t partial_metadata_ready_semaphore_id =
        get_named_compile_time_arg_val("partial_metadata_ready_semaphore_id");
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t previous_chunk_sent_semaphore_id =
        get_named_compile_time_arg_val("previous_chunk_sent_semaphore_id");
    constexpr uint32_t combine_sync_semaphore_id = get_named_compile_time_arg_val("combine_sync_semaphore_id");

    // When compute_only=1, the fused selective_reduce_combine path is bypassed and no combine
    // kernels run on combine cores. Skip the metadata-ready signal to combine cores.
    constexpr bool compute_only = get_named_compile_time_arg_val("compute_only") == 1;

    Semaphore<> partial_metadata_ready_sem(partial_metadata_ready_semaphore_id);
    Semaphore<> metadata_ready_sem(metadata_ready_semaphore_id);
    Semaphore<> previous_chunk_sent_sem(previous_chunk_sent_semaphore_id);

    // Device 2.0 migration: legacy primitives retained: these raw L1 semaphore addresses are
    // used as bases for multicast destinations (set_multicast / get_safe_multicast_noc_addr /
    // get_noc_addr) and for inter-core semaphore inc with precomposed uint64_t addresses.
    uint32_t metadata_ready_semaphore_addr = get_semaphore(metadata_ready_semaphore_id);
    const uint32_t combine_sync_addr = get_semaphore(combine_sync_semaphore_id);

    // Noc typed wrapper (reader uses default noc_index)
    Noc noc_obj(noc_index);

    // CircularBuffer typed wrappers
    CircularBuffer cb_indices_tensor(indices_tensor_cb_id);
    CircularBuffer cb_mapping_tensor(mapping_tensor_cb_id);
    CircularBuffer cb_scores_tensor(scores_tensor_cb_id);
    CircularBuffer cb_tilize_input(tilize_input_cb_id);
    CircularBuffer cb_e_t(e_t_cb_id);
    CircularBuffer cb_expert_activation(expert_activation_cb_id);
    CircularBuffer cb_per_expert_total_tokens(per_expert_total_tokens_cb_id);
    CircularBuffer cb_total_chunks(total_chunks_cb_id);
    CircularBuffer cb_brisc_e_t(brisc_e_t_cb_id);
    CircularBuffer cb_brisc_expert_counts(brisc_expert_counts_cb_id);
    CircularBuffer cb_brisc_expert_activation(brisc_expert_activation_cb_id);
    CircularBuffer cb_brisc_activated_count(brisc_activated_count_cb_id);
    CircularBuffer cb_remote_counts(remote_counts_cb_id);

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);                     // 0
    [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 1 - not used by reader
    [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 2 - not used by reader
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);                   // 3
    uint32_t per_expert_total_tokens_output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 4
    uint32_t expert_activation_output_address = get_arg_val<uint32_t>(rt_args_idx++);               // 5
    uint32_t e_t_output_address = get_arg_val<uint32_t>(rt_args_idx++);                             // 6
    bool is_drain_tilize_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);                         // 7
    [[maybe_unused]] bool is_secondary_mcaster = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 8 - not used by reader
    [[maybe_unused]] uint32_t initial_mcast_gather_core_nox_x =
        get_arg_val<uint32_t>(rt_args_idx++);  // 9 - not used by reader
    [[maybe_unused]] uint32_t initial_mcast_gather_core_nox_y =
        get_arg_val<uint32_t>(rt_args_idx++);                                // 10 - not used by reader
    uint32_t global_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);  // 11
    [[maybe_unused]] uint32_t mcast_group_subtoken_offset =
        get_arg_val<uint32_t>(rt_args_idx++);  // 12 - not used by reader
    [[maybe_unused]] uint32_t mcast_group_subtoken_size =
        get_arg_val<uint32_t>(rt_args_idx++);                          // 13 - not used by reader
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);     // 14
    uint32_t core_token_start = get_arg_val<uint32_t>(rt_args_idx++);  // 15
    uint32_t core_token_end = get_arg_val<uint32_t>(rt_args_idx++);    // 16
    uint32_t tilize_core_idx = get_arg_val<uint32_t>(rt_args_idx++);   // 17

    // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output, expert_activation_output
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto per_expert_total_tokens_output_args =
        TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto expert_activation_output_args =
        TensorAccessorArgs<per_expert_total_tokens_output_args.next_compile_time_args_offset()>();
    constexpr auto e_t_output_args =
        TensorAccessorArgs<expert_activation_output_args.next_compile_time_args_offset()>();

    // TensorAccessors
    const auto input_tensor_addr_gen = TensorAccessor(input_args, input_tensor_address);
    // indices not used by reader
    // scores not used by reader
    const auto mapping_tensor_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address);
    [[maybe_unused]] const auto per_expert_total_tokens_output_tensor_addr_gen =
        TensorAccessor(per_expert_total_tokens_output_args, per_expert_total_tokens_output_tensor_address);
    const auto expert_activation_output_tensor_addr_gen =
        TensorAccessor(expert_activation_output_args, expert_activation_output_address);
    const auto e_t_output_tensor_addr_gen = TensorAccessor(e_t_output_args, e_t_output_address);

    // Constants
    constexpr uint32_t one_page = 1;

    // Size of e_t buffer for all experts (for multicast)
    constexpr uint32_t e_t_buffer_total_size = experts_per_device * (tokens + 1) * e_t_entry_size;

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    [[maybe_unused]] constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;
    constexpr uint32_t tokens_per_device = tokens / dispatch_devices;

    // Aligned row size for expert_activation buffer (in bytes)
    constexpr uint32_t aligned_activation_row_bytes =
        ((2 * experts_per_device + 1) * sizeof(uint32_t) + l1_alignment - 1) / l1_alignment * l1_alignment;

    // NOC coordinates for all tilize cores (for cross-core communication)
    // Runtime args 11+: [core0_noc_x, core0_noc_y, core1_noc_x, core1_noc_y, ...]
    uint32_t tilize_noc_x[num_tilize_cores];
    uint32_t tilize_noc_y[num_tilize_cores];
    for (uint32_t i = 0; i < num_tilize_cores; i++) {
        tilize_noc_x[i] = get_arg_val<uint32_t>(rt_args_idx++);
        tilize_noc_y[i] = get_arg_val<uint32_t>(rt_args_idx++);
    }

    // Read the mapping tensor page for this device (linearized_mesh_coord)
    // This gives us the expert -> device mapping from this device's perspective
    // Reserve all pages (tokens)
    cb_mapping_tensor.reserve_back(mapping_pages);
    for (uint32_t i = 0; i < mapping_pages; i++) {
        noc_obj.async_read(
            mapping_tensor_addr_gen,
            cb_mapping_tensor,
            aligned_mapping_page_size,
            {.page_id = i},
            {.offset_bytes = i * aligned_mapping_page_size});
    }

    cb_expert_activation.reserve_back(tokens);
    init_expert_activation_buffer_async<selected_experts_k, tokens, experts_per_device, l1_alignment>(
        noc_obj, cb_expert_activation);

    // ========== NON-DRAIN CORES: Read indices/scores from drain core ==========
    // IMPORTANT: This must happen BEFORE pushing mapping_tensor_cb_id, since BRISC waits on that
    // and will immediately start reading indices/scores. Race condition otherwise!
    // Non-drain cores need to fetch their portion of the indices/scores tensors from drain core's L1 shard
    if (!is_drain_tilize_core) {
        // Calculate the byte offset for this core's token range within the indices/scores buffers
        uint32_t token_byte_offset_indices = core_token_start * aligned_indices_page_size;
        uint32_t token_byte_offset_scores = core_token_start * aligned_scores_page_size;
        uint32_t num_tokens_this_core = core_token_end - core_token_start;

        // Get local CB addresses (same relative address as drain core)
        uint32_t local_indices_addr = cb_indices_tensor.get_read_ptr() + token_byte_offset_indices;
        uint32_t local_scores_addr = cb_scores_tensor.get_write_ptr() + token_byte_offset_scores;

        // Calculate drain core's source addresses
        // Note: CB addresses are allocated at same L1 offset on all cores, so we use local get_read_ptr
        // and apply it to drain core's NOC address
        uint64_t drain_indices_noc_addr = get_noc_addr(drain_core_noc_x, drain_core_noc_y, local_indices_addr);
        uint64_t drain_scores_noc_addr = get_noc_addr(drain_core_noc_x, drain_core_noc_y, local_scores_addr);

        // NOC read indices and scores for this core's token range

        if (num_tokens_this_core > 0) {
            // Device 2.0 migration: legacy primitive retained: src is a precomposed uint64_t
            // cross-core NoC address from get_noc_addr(x,y,addr)
            noc_async_read(
                drain_indices_noc_addr, local_indices_addr, num_tokens_this_core * aligned_indices_page_size);
            // Device 2.0 migration: legacy primitive retained: src is a precomposed uint64_t
            // cross-core NoC address from get_noc_addr(x,y,addr)
            noc_async_read(drain_scores_noc_addr, local_scores_addr, num_tokens_this_core * aligned_scores_page_size);
        }
    }

    // Wait for all reads to complete (mapping + indices/scores for non-drain)
    noc_obj.async_read_barrier();

    // Now safe to signal BRISC - all data is in place
    cb_mapping_tensor.push_back(mapping_pages);
    cb_expert_activation.push_back(tokens);

    // DEBUG: print_expert_activation_buffer<experts_per_device, l1_alignment>(cb_expert_activation, 0, tokens);

    // Get pointer to the mapping data
    uint16_t* expert_to_device_map = reinterpret_cast<uint16_t*>(
        cb_mapping_tensor.get_read_ptr() + linearized_mesh_coord * aligned_mapping_page_size);
    uint16_t local_expert_ids[experts_per_device];
    uint32_t local_expert_count = 0;
    for (uint32_t i = 0; i < experts; i++) {
        uint16_t expert_mesh_coord = expert_to_device_map[i];
        if (expert_mesh_coord == linearized_mesh_coord) {
            if (local_expert_count >= experts_per_device) {
                // DEBUG: DPRINT("Error: more than {} experts on device {}\n", experts_per_device,
                // linearized_mesh_coord);
                ASSERT(false);
            }
            // DEBUG: DPRINT("Device {} : Local expert {} is {}\n", linearized_mesh_coord, local_expert_count, i);

            local_expert_ids[local_expert_count] = i;
            local_expert_count++;
        }
    }

    // Pre-compute base addresses (avoid repeated calls in hot loop)
    const uint32_t mapping_base = cb_mapping_tensor.get_read_ptr();
    const uint32_t e_t_buffer_base = cb_e_t.get_write_ptr();

    // Array to hold per-expert token counts (filled by all cores now)
    uint32_t num_activated_tokens_per_expert[experts_per_device] = {0};

    // ========== ALL CORES: Build e_t buffer and per-expert counts for their token range ==========
    // Each core processes a portion of total tokens: [core_token_start, core_token_end)
    // Within each core: NCRISC processes first half, BRISC processes second half
    uint32_t tokens_this_core = core_token_end - core_token_start;
    uint32_t ncrisc_token_start = core_token_start;
    uint32_t ncrisc_token_end = core_token_start + tokens_this_core / 2;
    uint32_t brisc_tokens_capacity = tokens_this_core / 2;

    // indices/scores are in CB - drain has shard, non-drain read via NOC in Step 2
    uint32_t num_activated_tokens = 0;

    const uint32_t indices_base = cb_indices_tensor.get_read_ptr();
    const uint32_t scores_base = cb_scores_tensor.get_read_ptr();
    const uint32_t expert_activation_base = cb_expert_activation.get_write_ptr();

    // Cache source_device_mapping - only changes every tokens_per_device tokens
    // Reduces mapping loads from 512 to 16 (dispatch_devices)
    uint32_t prev_device_in_group = UINT32_MAX;
    const uint16_t* source_device_mapping = nullptr;

    // NCRISC processes first half of this core's tokens
    for (uint32_t t = ncrisc_token_start; t < ncrisc_token_end; t++) {
        // source_device only changes every tokens_per_device tokens
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

        // Defer pointer calculation until we know token is activated
        uint32_t* expert_activation_l1_ptr = nullptr;
        bool activated = false;

        for (uint32_t k = 0; k < selected_experts_k; k++) {
            const uint16_t selected_expert = token_indices[k];

            // Check if this expert maps to our device first (likely to fail, skip early)
            if (source_device_mapping[selected_expert] != linearized_mesh_coord) {
                continue;
            }

            // Now check if it's one of our local experts
            for (uint32_t e = 0; e < local_expert_count; e++) {
                if (selected_expert == local_expert_ids[e]) {
                    // First activation for this token - set up pointer and write token id
                    if (!activated) {
                        expert_activation_l1_ptr = reinterpret_cast<uint32_t*>(
                            expert_activation_base + num_activated_tokens * aligned_activation_row_bytes);
                        expert_activation_l1_ptr[0] = t;
                        activated = true;
                    }

                    // Write k-index and score for this expert
                    expert_activation_l1_ptr[1 + e] = k;
                    expert_activation_l1_ptr[1 + experts_per_device + e] = static_cast<uint32_t>(token_scores[k]);

                    // Write to e_t buffer (16B aligned entries for NOC DMA compatibility)
                    const uint32_t e_t_offset =
                        (e * (tokens + 1) + num_activated_tokens_per_expert[e]) * e_t_entry_size;
                    *reinterpret_cast<uint32_t*>(e_t_buffer_base + e_t_offset) = t;
                    num_activated_tokens_per_expert[e]++;

                    break;  // Each k can only match one local expert, no need to check others
                }
            }
        }

        if (activated) {
            num_activated_tokens++;
        }
    }

    // ========== ALL CORES: WAIT FOR BRISC AND MERGE ==========
    // Wait for BRISC to finish processing second half and push its counts
    cb_brisc_expert_counts.wait_front(one_page);
    volatile tt_l1_ptr uint32_t* brisc_counts =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_brisc_expert_counts.get_read_ptr());

    // Wait for BRISC's e_t buffer to be ready
    cb_brisc_e_t.wait_front(one_page);
    const uint32_t brisc_e_t_buffer_base = cb_brisc_e_t.get_read_ptr();

    // Wait for BRISC's expert_activation buffer and count
    cb_brisc_activated_count.wait_front(one_page);
    uint32_t brisc_activated_count =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_brisc_activated_count.get_read_ptr());
    cb_brisc_expert_activation.wait_front(one_page);
    const uint32_t brisc_expert_activation_base = cb_brisc_expert_activation.get_read_ptr();

    // Merge BRISC's e_t buffer entries into main e_t buffer using NOC DMA
    // For each expert: copy BRISC's tokens after NCRISC's tokens
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t ncrisc_count = num_activated_tokens_per_expert[e];
        uint32_t brisc_count = brisc_counts[e];

        if (brisc_count > 0) {
            // Source: BRISC's e_t buffer for expert e (16B aligned entries)
            uint32_t brisc_e_t_src_addr = brisc_e_t_buffer_base + e * brisc_tokens_capacity * e_t_entry_size;

            // Destination: main e_t buffer, after NCRISC's entries (16B aligned entries)
            uint32_t e_t_dst_addr = e_t_buffer_base + (e * (tokens + 1) + ncrisc_count) * e_t_entry_size;

            // Use NOC DMA for L1-to-L1 copy (local loopback)
            // Device 2.0 migration: legacy primitive retained: get_noc_addr(local_l1_addr) is used
            // to form a self-loopback NoC source for local L1->L1 copy
            uint64_t src_noc_addr = get_noc_addr(brisc_e_t_src_addr);
            noc_async_read(src_noc_addr, e_t_dst_addr, brisc_count * e_t_entry_size);
        }

        // Update total count for this expert
        num_activated_tokens_per_expert[e] = ncrisc_count + brisc_count;
    }

    // Merge BRISC's expert_activation buffer using NOC DMA
    // Copy all of BRISC's activated rows after NCRISC's activated rows
    if (brisc_activated_count > 0) {
        uint32_t expert_activation_dst_addr =
            expert_activation_base + num_activated_tokens * aligned_activation_row_bytes;
        // Device 2.0 migration: legacy primitive retained: get_noc_addr(local_l1_addr) is used
        // to form a self-loopback NoC source for local L1->L1 copy
        uint64_t brisc_activation_src_noc_addr = get_noc_addr(brisc_expert_activation_base);
        noc_async_read(
            brisc_activation_src_noc_addr,
            expert_activation_dst_addr,
            brisc_activated_count * aligned_activation_row_bytes);
    }

    // Wait for all NOC DMA copies to complete
    noc_obj.async_read_barrier();

    // Update total activated token count to include BRISC's tokens
    num_activated_tokens += brisc_activated_count;

    // Pop BRISC's CBs (cleanup)
    cb_brisc_expert_counts.pop_front(one_page);
    cb_brisc_e_t.pop_front(one_page);
    cb_brisc_expert_activation.pop_front(one_page);
    cb_brisc_activated_count.pop_front(one_page);

    // ========== CROSS-CORE CONSOLIDATION (Steps 4-6) ==========
    // Non-drain cores: send counts to drain and wait for consolidated buffer
    // Drain core: receive from non-drain, consolidate, multicast
    if (is_drain_tilize_core) {
        // ========== Step 5: Drain receives counts from non-drain cores and consolidates ==========
        if (num_tilize_cores > 1) {
            // Wait for all non-drain cores to send their counts
            partial_metadata_ready_sem.wait(num_tilize_cores - 1);

            // Read counts from each non-drain core and consolidate their buffers
            // tilize_noc_x and tilize_noc_y arrays were populated from runtime args earlier
            const uint32_t remote_counts_base = cb_remote_counts.get_read_ptr();

            for (uint32_t core_idx = 1; core_idx < num_tilize_cores; core_idx++) {
                // Read this core's counts from remote_counts_cb
                uint32_t* remote_counts =
                    reinterpret_cast<uint32_t*>(remote_counts_base + (core_idx - 1) * remote_counts_entry_size);

                uint32_t remote_e_t_counts[experts_per_device];
                for (uint32_t e = 0; e < experts_per_device; e++) {
                    remote_e_t_counts[e] = remote_counts[e];
                }
                uint32_t remote_activated_count = remote_counts[experts_per_device];

                // Get remote core's NOC coordinates
                uint32_t remote_noc_x = tilize_noc_x[core_idx];
                uint32_t remote_noc_y = tilize_noc_y[core_idx];

                // Pull this core's e_t buffer entries for each expert
                // Remote core's e_t buffer is at the same CB address, entries start at offset 0 for each expert
                for (uint32_t e = 0; e < experts_per_device; e++) {
                    uint32_t remote_count = remote_e_t_counts[e];
                    if (remote_count > 0) {
                        // Source: remote core's e_t buffer, expert e's section starts at 0
                        uint32_t remote_e_t_addr = cb_e_t.get_write_ptr() + e * (tokens + 1) * e_t_entry_size;
                        uint64_t remote_e_t_noc_addr = get_noc_addr(remote_noc_x, remote_noc_y, remote_e_t_addr);

                        // Destination: drain's e_t buffer, after current entries for this expert
                        uint32_t local_e_t_dst =
                            e_t_buffer_base + (e * (tokens + 1) + num_activated_tokens_per_expert[e]) * e_t_entry_size;

                        // Device 2.0 migration: legacy primitive retained: src is a precomposed
                        // uint64_t cross-core NoC address from get_noc_addr(x,y,addr)
                        noc_async_read(remote_e_t_noc_addr, local_e_t_dst, remote_count * e_t_entry_size);

                        // Update drain's count for this expert
                        num_activated_tokens_per_expert[e] += remote_count;
                    }
                }

                // Pull this core's expert_activation buffer rows
                if (remote_activated_count > 0) {
                    // Source: remote core's expert_activation buffer, rows start at 0
                    uint32_t remote_activation_addr = cb_expert_activation.get_write_ptr();
                    uint64_t remote_activation_noc_addr =
                        get_noc_addr(remote_noc_x, remote_noc_y, remote_activation_addr);

                    // Destination: drain's expert_activation buffer, after current rows
                    uint32_t local_activation_dst =
                        expert_activation_base + num_activated_tokens * aligned_activation_row_bytes;

                    // Device 2.0 migration: legacy primitive retained: src is a precomposed
                    // uint64_t cross-core NoC address from get_noc_addr(x,y,addr)
                    noc_async_read(
                        remote_activation_noc_addr,
                        local_activation_dst,
                        remote_activated_count * aligned_activation_row_bytes);

                    // Update drain's total activated count
                    num_activated_tokens += remote_activated_count;
                }
            }

            // Wait for all consolidation reads to complete
            noc_obj.async_read_barrier();
        }

        // cap off e_t buffer with -1 (now includes merged counts, 16B aligned entries)
        for (uint32_t e = 0; e < experts_per_device; e++) {
            uint32_t e_t_buffer_addr = cb_e_t.get_write_ptr() + e * (tokens + 1) * e_t_entry_size;
            uint32_t* e_t_sentinel_ptr =
                reinterpret_cast<uint32_t*>(e_t_buffer_addr + num_activated_tokens_per_expert[e] * e_t_entry_size);
            *e_t_sentinel_ptr = static_cast<uint32_t>(-1);
        }

        // Push per-expert token counts to CB for writer to read
        cb_per_expert_total_tokens.reserve_back(1);
        uint32_t* per_expert_counts_ptr = reinterpret_cast<uint32_t*>(cb_per_expert_total_tokens.get_write_ptr());
        for (uint32_t e = 0; e < experts_per_device; e++) {
            per_expert_counts_ptr[e] = num_activated_tokens_per_expert[e];
        }
        cb_per_expert_total_tokens.push_back(1);

        cb_total_chunks.reserve_back(one_page);
        uint32_t total_chunks = 0;
        for (uint32_t e = 0; e < experts_per_device; e++) {
            total_chunks += (num_activated_tokens_per_expert[e] + tokens_per_chunk - 1) / tokens_per_chunk;
        }
        *reinterpret_cast<uint32_t*>(cb_total_chunks.get_write_ptr()) = total_chunks;
        cb_total_chunks.push_back(one_page);

        // ========== Write expert_activation buffer to DRAM ==========
        // Write to DRAM: activated rows (num_activated_tokens) rows
        uint32_t expert_activation_write_size = num_activated_tokens * aligned_activation_row_bytes;
        uint64_t expert_activation_dram_addr = expert_activation_output_tensor_addr_gen.get_noc_addr(0);
        if (num_activated_tokens > 0) {
            // Device 2.0 migration: legacy primitive retained: dst is a precomposed uint64_t DRAM
            // NoC address from get_noc_addr(0, accessor)
            noc_async_write(expert_activation_base, expert_activation_dram_addr, expert_activation_write_size);
        }
        // Barrier for this write is at the very end of the kernel

        // DEBUG: print_e_t_buffer<experts_per_device, tokens, e_t_entry_size>(cb_e_t);

        // Multicast e_t buffer, per_expert_counts, and total_chunks to non-drain-sync cores
        if (num_tilize_cores > 1) {
            uint32_t e_t_cb_read_ptr = cb_e_t.get_read_ptr();
            uint32_t per_expert_total_tokens_cb_read_ptr = cb_per_expert_total_tokens.get_read_ptr();
            uint32_t total_chunks_cb_read_ptr = cb_total_chunks.get_read_ptr();

            uint64_t e_t_mcast_addr = get_safe_multicast_noc_addr(
                tilize_mcast_start_x, tilize_mcast_start_y, tilize_mcast_end_x, tilize_mcast_end_y, e_t_cb_read_ptr);

            uint64_t per_expert_counts_mcast_addr = get_safe_multicast_noc_addr(
                tilize_mcast_start_x,
                tilize_mcast_start_y,
                tilize_mcast_end_x,
                tilize_mcast_end_y,
                per_expert_total_tokens_cb_read_ptr);

            uint64_t total_chunks_mcast_addr = get_safe_multicast_noc_addr(
                tilize_mcast_start_x,
                tilize_mcast_start_y,
                tilize_mcast_end_x,
                tilize_mcast_end_y,
                total_chunks_cb_read_ptr);

            // Multicast e_t buffer to all tilize cores
            // Device 2.0 migration: legacy primitive retained: noc_async_write_multicast is invoked
            // with precomposed uint64_t multicast destination address; Noc::async_write_multicast does
            // not take a raw uint64_t multicast destination
            noc_async_write_multicast(
                e_t_cb_read_ptr, e_t_mcast_addr, e_t_buffer_total_size, tilize_bounding_box_num_cores - 1);

            // Multicast per_expert_counts to all tilize cores
            // Device 2.0 migration: legacy primitive retained: see above
            noc_async_write_multicast(
                per_expert_total_tokens_cb_read_ptr,
                per_expert_counts_mcast_addr,
                experts_per_device * sizeof(uint32_t),
                tilize_bounding_box_num_cores - 1);

            // Multicast total_chunks to all tilize cores
            // Device 2.0 migration: legacy primitive retained: see above
            noc_async_write_multicast(
                total_chunks_cb_read_ptr, total_chunks_mcast_addr, sizeof(uint32_t), tilize_bounding_box_num_cores - 1);

            // Signal non-drain cores via semaphore multicast
            // First, set the local semaphore to 1 - this is the value that will be multicast
            metadata_ready_sem.set(1);

            uint64_t semaphore_mcast_addr = get_safe_multicast_noc_addr(
                tilize_mcast_start_x,
                tilize_mcast_start_y,
                tilize_mcast_end_x,
                tilize_mcast_end_y,
                metadata_ready_semaphore_addr);

            // Flush writes since we change the local value of metadata_ready_semaphore when signalling
            // to the matmul cores (vs here where we signal to the non-drain-sync tilize cores )
            noc_obj.async_writes_flushed();

            // Multicast the value 1 to all non-drain tilize cores
            // Device 2.0 migration: legacy primitive retained: multicast loopback semaphore set with
            // precomposed uint64_t multicast destination (get_safe_multicast_noc_addr) and raw local
            // sem address as source
            noc_semaphore_set_multicast(
                metadata_ready_semaphore_addr, semaphore_mcast_addr, tilize_bounding_box_num_cores - 1);
        }

        volatile tt_l1_ptr uint32_t* num_tokens_per_expert =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_per_expert_total_tokens.get_read_ptr());

        // Multicast the per-expert token counts to ALL worker cores (tilize + matmul + combine)
        // this might be overkill, only matmul and combine need this right now but we can just do one big MC
        uint64_t all_worker_cores_expert_counts_mcast_addr = get_safe_multicast_noc_addr(
            all_worker_cores_mcast_start_x,
            all_worker_cores_mcast_start_y,
            all_worker_cores_mcast_end_x,
            all_worker_cores_mcast_end_y,
            cb_per_expert_total_tokens.get_read_ptr());

        // Device 2.0 migration: legacy primitive retained: noc_async_write_multicast with precomposed
        // uint64_t multicast destination
        noc_async_write_multicast(
            cb_per_expert_total_tokens.get_read_ptr(),
            all_worker_cores_expert_counts_mcast_addr,
            per_expert_total_tokens_output_page_size,
            all_worker_cores_bounding_box_num_cores - 1);  // Exclude self

        // Ensure multicast completes before signaling semaphore
        noc_obj.async_write_barrier();

        // Signal readiness via semaphore
        metadata_ready_sem.set(1);

        // get mcast address for semaphore
        uint64_t matmul_metadata_ready_semaphore_mcast_addr = get_safe_multicast_noc_addr(
            matmul_mcast_start_x,
            matmul_mcast_start_y,
            matmul_mcast_end_x,
            matmul_mcast_end_y,
            metadata_ready_semaphore_addr);

        // multicast semaphore
        // Device 2.0 migration: legacy primitive retained: multicast loopback semaphore set with
        // precomposed uint64_t multicast destination
        noc_semaphore_set_multicast(
            metadata_ready_semaphore_addr, matmul_metadata_ready_semaphore_mcast_addr, matmul_bounding_box_num_cores);
    }  // End of is_drain_tilize_core block
    else {
        // ========== NON-DRAIN tilize CORE: Step 4 - Send counts to drain ==========
        // Each non-drain core writes its counts to drain core's remote_counts_cb
        // Layout in remote_counts_cb: [core1_counts][core2_counts][core3_counts]...
        // Each entry: [e_t_count_expert0, e_t_count_expert1, ..., activated_count] (16B aligned)

        // Calculate offset in drain's remote_counts_cb for this core
        // tilize_core_idx is 1, 2, 3... for non-drain cores, so offset is (idx-1) * entry_size
        uint32_t remote_counts_offset = (tilize_core_idx - 1) * remote_counts_entry_size;

        // Get drain core's remote_counts_cb address
        uint32_t local_counts_addr = cb_remote_counts.get_read_ptr();
        uint64_t drain_counts_noc_addr =
            get_noc_addr(drain_core_noc_x, drain_core_noc_y, local_counts_addr + remote_counts_offset);

        // Pack counts into local buffer first (we can use the local remote_counts_cb space temporarily)
        uint32_t* counts_ptr = reinterpret_cast<uint32_t*>(local_counts_addr);
        for (uint32_t e = 0; e < experts_per_device; e++) {
            counts_ptr[e] = num_activated_tokens_per_expert[e];
        }
        counts_ptr[experts_per_device] = num_activated_tokens;

        // Write counts to drain core's remote_counts_cb
        // Device 2.0 migration: legacy primitive retained: dst is a precomposed uint64_t
        // cross-core NoC address from get_noc_addr(x,y,addr)
        noc_async_write(local_counts_addr, drain_counts_noc_addr, remote_counts_entry_size);
        noc_obj.async_write_barrier();

        // Signal drain core via semaphore increment
        partial_metadata_ready_sem.up(noc_obj, drain_core_noc_x, drain_core_noc_y, 1);

        // ========== NON-DRAIN tilize CORE: Wait for drain core to multicast data ==========
        // Wait for the semaphore signal from drain core
        metadata_ready_sem.wait(1);

        // Read per-expert counts from the CB (multicast by drain core)
        // The data was written directly to our CB by the multicast
        volatile tt_l1_ptr uint32_t* per_expert_counts =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_per_expert_total_tokens.get_read_ptr());
        for (uint32_t e = 0; e < experts_per_device; e++) {
            num_activated_tokens_per_expert[e] = per_expert_counts[e];
        }

        // Push per_expert_total_tokens_cb and total_chunks_cb so writer can read them
        // (drain core already pushed, non-drain cores need to mark as available)
        cb_per_expert_total_tokens.reserve_back(1);
        cb_per_expert_total_tokens.push_back(1);
        cb_total_chunks.reserve_back(one_page);
        cb_total_chunks.push_back(one_page);
    }

    if (is_drain_tilize_core) {
        // write out e_t_output_tensor
        for (uint32_t e = 0; e < experts_per_device; ++e) {
            noc_obj.async_write(
                cb_e_t,
                e_t_output_tensor_addr_gen,
                e_t_output_page_size,
                {.offset_bytes = e * e_t_output_page_size},
                {.page_id = e});
        }

        noc_obj.async_write_barrier();

        // signal to A2A combine that metadata is available. Separate signal from matmul because e_t write is also
        // needed. Skipped in compute_only mode (no combine kernels listening).
        if constexpr (!compute_only) {
            // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
            // (combine_sync target on a different core) cannot be wrapped by Semaphore<>::inc which
            // binds to a per-program id
            const uint64_t combine_sync_noc_addr =
                safe_get_noc_addr(combine_sync_noc_x, combine_sync_noc_y, combine_sync_addr, 1);
            noc_semaphore_inc(combine_sync_noc_addr, 1);

            noc_obj.async_atomic_barrier();
        }
    }

    // DEBUG
    // print_e_t_buffer<experts_per_device, tokens, e_t_entry_size>(cb_e_t);
    // print_expert_activation_buffer<experts_per_device, l1_alignment>(cb_expert_activation, 0,
    // std::max(num_activated_tokens_per_expert[0], num_activated_tokens_per_expert[1]));

    // ========== ALL CORES: Read activated tokens from sparse buffer and pack into tilize input CB ==========
    // The e_t buffer contains sparse token IDs for each expert, with 16B aligned entries
    uint32_t num_chunks_sent = 0;
    for (uint32_t e = 0; e < experts_per_device; e++) {
        uint32_t num_tokens = num_activated_tokens_per_expert[e];
        uint32_t e_t_expert_addr = e_t_buffer_base + e * (tokens + 1) * e_t_entry_size;

        // Process tokens in chunks of tokens_per_chunk
        for (uint32_t chunk_start = 0; chunk_start < num_tokens; chunk_start += tokens_per_chunk) {
            uint32_t tokens_in_chunk = std::min(tokens_per_chunk, num_tokens - chunk_start);

            cb_tilize_input.reserve_back(tokens_per_chunk);

            // Read each activated token from the sparse input buffer
            for (uint32_t i = 0; i < tokens_in_chunk; i++) {
                // Get sparse token ID from e_t buffer (16B aligned entries)
                uint32_t token_id = *reinterpret_cast<uint32_t*>(e_t_expert_addr + (chunk_start + i) * e_t_entry_size);
                // read the token from the input tensor at the tilize subtoken offset and size
                noc_obj.async_read(
                    input_tensor_addr_gen,
                    cb_tilize_input,
                    subtoken_size,
                    {.page_id = token_id, .offset_bytes = global_subtoken_offset},
                    {.offset_bytes = i * subtoken_size});
            }
            noc_obj.async_read_barrier();
            cb_tilize_input.push_back(tokens_per_chunk);  // Push full chunk (padding is garbage, that's OK)
            num_chunks_sent++;

            // Wait until previous chunk arrives on the matmul cores before reading in another chunk of tokens.
            // Since both the reader and writer use NoC1, we want writer to have priority access so that chunks
            // arrive at the matmul cores earlier. Also, to do linked mcast transactions we need NoC to be completely
            // idle during mcast. The very last wait is technically redundant since we won't be reading in another chunk
            // of tokens, however it's still required so we don't use NoC1 to write out the output tensors until the
            // last linked mcast completes.
            previous_chunk_sent_sem.wait(num_chunks_sent);
        }
    }
}
