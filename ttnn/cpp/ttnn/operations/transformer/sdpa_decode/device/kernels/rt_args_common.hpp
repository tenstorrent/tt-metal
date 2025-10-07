// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
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

inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_runtime_args(
    int cur_pos,
    int cur_batch,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    std::optional<uint32_t> sliding_window = std::nullopt) {
    uint32_t window_start = 0;
    uint32_t window_start_unaligned = 0;  // Keep track of the actual window start for masking
    uint32_t valid_seq_len;

    if (sliding_window.has_value() && sliding_window.value() > 0) {
        // Calculate actual window bounds
        uint32_t window_end = cur_pos;  // exclusive end
        window_start_unaligned = (window_end > sliding_window.value()) ? (window_end - sliding_window.value()) : 0;

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
    uint32_t total_chunks = valid_seq_len / k_chunk_size;
    uint32_t active_chunks = total_chunks - window_start_chunk;

    uint32_t k_chunk_start = window_start_chunk;
    uint32_t k_chunk_end = window_start_chunk;

    // Distribute active chunks among cores
    if (num_cores_per_batch > int(active_chunks)) {
        int chunks_per_core = (core_num < int(active_chunks)) ? 1 : 0;
        k_chunk_start = window_start_chunk + (active_chunks - core_num - 1) * chunks_per_core;
        k_chunk_end = window_start_chunk + (active_chunks - core_num) * chunks_per_core;
    } else {
        int chunks_per_core = active_chunks / num_cores_per_batch;
        int residuals = active_chunks % num_cores_per_batch;
        int reversed_core_num = num_cores_per_batch - core_num - 1;
        k_chunk_start =
            window_start_chunk + reversed_core_num * chunks_per_core + std::min(residuals, reversed_core_num);
        k_chunk_end = k_chunk_start + chunks_per_core;
        if (reversed_core_num < residuals) {
            k_chunk_end += 1;
        }
    }
    return {pst_value, active_chunks, k_chunk_start, k_chunk_end, window_start_unaligned};
}

// Backward compatibility overload for compute kernel (no sliding window support)
// inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_runtime_args(
//     int cur_pos, int cur_batch, int core_num, int num_cores_per_batch, uint32_t k_chunk_size) {
//     auto [pst_value, active_chunks, k_chunk_start, k_chunk_end, window_start_unaligned] =
//         get_runtime_args(cur_pos, cur_batch, core_num, num_cores_per_batch, k_chunk_size, std::nullopt);
//     return {pst_value, active_chunks, k_chunk_start, k_chunk_end};
// }

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
