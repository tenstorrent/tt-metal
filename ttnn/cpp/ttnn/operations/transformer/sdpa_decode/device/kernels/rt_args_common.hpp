// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <optional>
#include <tuple>

/******************************************************************************
 *                   Tree Reduction Helper Functions                          *
 ******************************************************************************/

// Maximum number of reduction rounds (supports up to 2^MAX_TREE_REDUCTION_ROUNDS cores)
constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS = 6;  // Supports up to 64 cores per reduction group

/**
 * Count trailing zeros in a number (position of lowest set bit)
 * Returns bit_width if n == 0
 */
inline uint32_t count_trailing_zeros(uint32_t n) {
    if (n == 0) {
        return 32;
    }
    uint32_t count = 0;
    while ((n & 1) == 0) {
        n >>= 1;
        count++;
    }
    return count;
}

/**
 * Calculate ceil(log2(n))
 */
inline uint32_t ceil_log2(uint32_t n) {
    if (n <= 1) {
        return 0;
    }
    uint32_t log = 0;
    n--;
    while (n > 0) {
        n >>= 1;
        log++;
    }
    return log;
}

/**
 * Tree reduction structure for a single core
 *
 * In binary tree reduction with N cores (0 to N-1):
 * - Round r: cores where (core_id & ((1 << (r+1)) - 1)) == (1 << r) - 1 + (1 << r)
 *   receive from (core_id - (1 << r))
 *
 * Example for 8 cores:
 * - Round 0: 1 receives from 0, 3 receives from 2, 5 receives from 4, 7 receives from 6
 * - Round 1: 3 receives from 1, 7 receives from 5
 * - Round 2: 7 receives from 3, 7 is root
 */
struct TreeReductionParams {
    uint32_t num_rounds;                                     // Total rounds = ceil(log2(num_cores))
    uint32_t my_active_rounds;                               // How many rounds this core participates in
    bool is_root;                                            // Whether this core is the final reducer
    uint32_t parent_core_in_group;                           // Which core to send to (-1 if root)
    uint32_t send_at_round;                                  // At which round this core sends to parent (-1 if root)
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];  // Child core index at each round (-1 if none)
    uint32_t num_children;                                   // Total number of children across all rounds
};

/**
 * Compute tree reduction parameters for a given core
 *
 * @param core_id_in_group The core's index within the reduction group (0 to num_cores-1)
 * @param num_cores_in_group Total number of cores in the reduction group
 * @return TreeReductionParams structure with all tree info
 */
inline TreeReductionParams get_tree_reduction_params(uint32_t core_id_in_group, uint32_t num_cores_in_group) {
    TreeReductionParams params;

    // Initialize
    params.num_rounds = ceil_log2(num_cores_in_group);
    params.is_root = false;
    params.parent_core_in_group = UINT32_MAX;
    params.send_at_round = UINT32_MAX;
    params.num_children = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; r++) {
        params.children_per_round[r] = UINT32_MAX;
    }

    if (num_cores_in_group <= 1) {
        // Single core case - it's the root with no children
        params.is_root = true;
        params.my_active_rounds = 0;
        return params;
    }

    // Determine when this core sends to its parent
    // A core sends at round r if bit r is 0 and all lower bits are 1
    // i.e., the trailing 1s pattern: core_id ends with (r) 1s followed by 0
    uint32_t trailing_ones = count_trailing_zeros(~core_id_in_group);  // Count trailing 1s

    // Root is the last core (num_cores - 1)
    uint32_t root_core = num_cores_in_group - 1;

    // Find children at each round
    params.my_active_rounds = 0;
    for (uint32_t r = 0; r < params.num_rounds; r++) {
        uint32_t step = 1u << r;

        // A core receives at round r if:
        // - (core_id & step) != 0 (bit r is set)
        // - All bits below r are also set: (core_id & (step - 1)) == (step - 1)
        // This means core_id ends with (r+1) 1-bits

        uint32_t mask = (step << 1) - 1;  // mask for bits 0..r
        uint32_t expected = mask;         // all 1s in bits 0..r means we receive

        if ((core_id_in_group & mask) == expected) {
            // We receive from (core_id - step) at round r
            uint32_t child = core_id_in_group - step;
            if (child < num_cores_in_group) {
                params.children_per_round[r] = child;
                params.num_children++;
            }
            params.my_active_rounds = r + 1;
        }
    }

    // Determine when this core sends (if not root)
    if (core_id_in_group == root_core) {
        params.is_root = true;
        params.parent_core_in_group = UINT32_MAX;
        params.send_at_round = UINT32_MAX;
    } else {
        // Find the first round where this core doesn't receive
        uint32_t send_round = trailing_ones;
        uint32_t step = 1u << send_round;
        params.parent_core_in_group = core_id_in_group + step;
        params.send_at_round = send_round;

        // If parent is out of bounds, adjust
        if (params.parent_core_in_group >= num_cores_in_group) {
            // This shouldn't happen in a properly constructed tree
            // But handle edge case for non-power-of-2 cores
            params.is_root = true;
            params.parent_core_in_group = UINT32_MAX;
            params.send_at_round = UINT32_MAX;
        }
    }

    return params;
}

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

inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_runtime_args(
    int cur_pos,
    int cur_batch,
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
