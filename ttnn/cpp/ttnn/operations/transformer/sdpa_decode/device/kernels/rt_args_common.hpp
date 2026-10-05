// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/tree_reduction.hpp"
#include <optional>
#include <tuple>

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

template <uint8_t max>
inline uint8_t nearest_pow_of_2_up_to_8(uint32_t x) {
    if (x == 0) {
        return 1;  // Handle edge case when x is 0
    }

    // Round up to nearest power of 2
    --x;          // Decrease x by 1 to handle exact powers of 2
    x |= x >> 1;  // Propagate the highest set bit
    x |= x >> 2;  // Propagate the highest set bit

    // Uncomment if want to support full range of uin32_t
    // x |= x >> 4;  // Propagate the highest set bit
    // x |= x >> 8;  // Propagate the highest set bit
    // x |= x >> 16;  // Propagate the highest set bit

    uint32_t result = x + 1;  // Add 1 to get the next power of 2

    // Cap the result at max
    return (result > max) ? max : result;
}

inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_workload_for_core(
    int cur_pos,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    std::optional<uint32_t> sliding_window_size = std::nullopt) {
    uint32_t window_start = 0;
    uint32_t window_start_unaligned = 0;  // Keep track of the actual window start for masking
    uint32_t valid_seq_len;

    if (sliding_window_size.has_value() && sliding_window_size.value() > 0) {
        // Calculate actual window bounds
        uint32_t window_end = cur_pos + 1;  // exclusive end
        window_start_unaligned =
            (window_end > sliding_window_size.value()) ? (window_end - sliding_window_size.value()) : 0;

        // Round window_start down to chunk boundary to ensure we capture the full window
        uint32_t window_start_aligned = (window_start_unaligned / k_chunk_size) * k_chunk_size;

        // Round window_end up to chunk boundary to ensure we capture the full window
        uint32_t window_end_aligned = nearest_n(window_end, k_chunk_size);

        // Calculate valid_seq_len based on the sliding window range
        valid_seq_len = window_end_aligned - window_start_aligned;
        window_start = window_start_aligned;  // Use aligned start for chunk calculations
    } else {
        // Standard behavior: process from beginning up to cur_pos
        valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
        window_start = 0;
        window_start_unaligned = 0;
    }

    uint32_t pst_value = valid_seq_len / tt::constants::TILE_HEIGHT;
    uint32_t window_start_chunk = window_start / k_chunk_size;
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    uint32_t k_chunk_start = window_start_chunk;
    uint32_t k_chunk_end = window_start_chunk;

    // Distribute active chunks among cores
    if (num_cores_per_batch > int(num_chunks_value)) {
        int chunks_per_core = (core_num < int(num_chunks_value)) ? 1 : 0;
        k_chunk_start = window_start_chunk + (num_chunks_value - core_num - 1) * chunks_per_core;
        k_chunk_end = window_start_chunk + (num_chunks_value - core_num) * chunks_per_core;
    } else {
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int reversed_core_num = num_cores_per_batch - core_num - 1;
        k_chunk_start =
            window_start_chunk + reversed_core_num * chunks_per_core + std::min(residuals, reversed_core_num);
        k_chunk_end = k_chunk_start + chunks_per_core;
        if (reversed_core_num < residuals) {
            k_chunk_end += 1;
        }
    }

    return {pst_value, num_chunks_value, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk};
}

template <uint32_t Sk_chunk_t, uint32_t max_size>
inline uint32_t get_dynamic_Sk_chunk_t(int cur_pos) {
    if constexpr (Sk_chunk_t == 0) {
        // Cur_pos + 1 for position, but -1 for divup, so cancels out
        // ie. divup(a, b) = (a - 1) / b + 1
        uint32_t seq_len_in_tiles = cur_pos / tt::constants::TILE_HEIGHT + 1;

        // Use nearest power of 2 to nicely divide total CB size which is some factor of max_size
        // Technically, should not be an issue but seeing PCC issues when using 3 tiles eg.
        // - Can switch to nearest tile if this is fixed
        return nearest_pow_of_2_up_to_8<max_size>(seq_len_in_tiles);
    }
    return Sk_chunk_t;
}

/******************************************************************************
 *       Active-row core allocation (tenstorrent/tt-metal#59300)              *
 ******************************************************************************/

// Which padded batch row, kv head and reduction rank a core serves when cores are dealt to the
// ACTIVE rows of a step instead of to the padded batch. ``active`` lists the rows whose cur_pos
// is not -1, in row order. The pool of ``num_cores`` is FIXED by the host independently of the
// padded batch (that is the whole point: the split must not change between graphs), so the
// assignment is a function of the active rows alone. C cores over R active rows give
// floor(C / R) cores per row, split evenly over the kv heads and capped at
// ``max_cores_per_head`` (program_config.max_cores_per_head_batch); each (row, head) group
// reduces over ``group_size`` ranks whose static core indices are contiguous from
// ``group_base``. Cores left over (index >= R * group) are idle.
//
// The host guarantees num_cores >= B * num_kv_heads, so cores_per_row >= num_kv_heads for any
// R <= B and every active row gets at least one core per head; the max(1, ...) below is only
// a guard against a pool the host did not validate (a row would otherwise be dropped silently).
struct ActiveRowAssignment {
    bool idle = true;
    uint32_t row = 0;
    uint32_t head = 0;
    uint32_t rank = 0;
    uint32_t group_size = 1;
    uint32_t group_base = 0;
};

template <uint32_t num_cores, uint32_t num_kv_heads, uint32_t max_cores_per_head>
inline ActiveRowAssignment assign_core_to_active_row(
    const uint32_t* active, uint32_t active_count, uint32_t core_index) {
    static_assert(num_kv_heads >= 1 && max_cores_per_head >= 1);
    ActiveRowAssignment a;
    if (active_count == 0) {
        return a;
    }
    const uint32_t cores_per_row = num_cores / active_count;
    uint32_t cores_per_head = cores_per_row / num_kv_heads;
    if (cores_per_head == 0) {
        cores_per_head = 1;
    }
    if (cores_per_head > max_cores_per_head) {
        cores_per_head = max_cores_per_head;
    }
    const uint32_t group = cores_per_head * num_kv_heads;
    const uint32_t slot = core_index / group;
    if (slot >= active_count) {
        return a;
    }
    const uint32_t within = core_index % group;
    a.idle = false;
    a.row = active[slot];
    a.head = within / cores_per_head;
    a.rank = within % cores_per_head;
    a.group_size = cores_per_head;
    a.group_base = slot * group + a.head * cores_per_head;
    return a;
}
