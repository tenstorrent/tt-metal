// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

namespace ttnn::prim {

/**
 * @brief Finds the largest power of two less than or equal to input value
 *
 * Algorithm: Uses bit manipulation with count-leading-zeros (clz) instruction
 * to find the position of the most significant bit, then creates a power of 2.
 *
 * @param x Input value (must be > 0 for meaningful results)
 * @return Largest power of 2 ≤ x (returns 0 if x == 0)
 *
 * Examples:
 * - largest_power_of_two(15) = 8  (2^3)
 * - largest_power_of_two(16) = 16 (2^4)
 * - largest_power_of_two(100) = 64 (2^6)
 */
uint32_t largest_power_of_two(uint32_t x) { return x == 0 ? 0 : (1U << (31 - __builtin_clz(x))); }

/**
 * @brief Finds optimal core configuration for multi-core TopK execution
 *
 * This function determines the best way to distribute TopK work across multiple cores
 * by analyzing memory constraints, core availability, and workload balance. It searches
 * for a configuration that maximizes parallelization while staying within hardware limits.
 *
 * Algorithm overview:
 * 1. Start with a conservative split size based on available cores and width
 * 2. Iteratively try larger split sizes (powers of 2) up to max_dim
 * 3. For each split size, calculate required cores and memory costs
 * 4. Verify that configuration fits within available cores and memory
 * 5. Find contiguous core arrangement that matches the requirement
 * 6. Return the first valid configuration found
 *
 * Memory cost model:
 * - Gather cost: Data movement between cores (2 * num_cores * tile_sizes)
 * - Local cost: Per-core memory usage (split_size/TILE_WIDTH * tile_sizes)
 * - Total must fit within L1 memory per core
 *
 * @param width Total width of the dimension being processed (in elements)
 * @param min_dim Minimum allowed split size (hardware constraint)
 * @param max_dim Maximum allowed split size (hardware constraint)
 * @param k Number of top elements to find
 * @param core_range Available core grid for parallel execution
 * @param l1_size L1 cache size per core (memory constraint)
 * @param value_tile_size Memory size of value tiles
 * @param index_tile_size Memory size of index tiles
 * @return Optional TopKCoreConfig with optimal settings, or nullopt if impossible
 */
std::optional<TopKCoreConfig> find_topk_core_config(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size) {
    // Calculate the maximum number of cores available in the core grid
    const auto max_cores =
        (core_range.end_coord.y - core_range.start_coord.y - 1) * (core_range.end_coord.x - core_range.start_coord.x);

    // Calculate conservative starting split size:
    // 1. Divide width by tile width to get number of tiles
    // 2. Divide by largest power-of-two <= max_cores for balanced distribution
    // 3. Convert back to elements by multiplying by tile width
    // This ensures we start with a split size that can utilize most available cores
    const uint32_t start_split_size =
        static_cast<uint32_t>(width / tt::constants::TILE_WIDTH / largest_power_of_two(max_cores)) *
        tt::constants::TILE_WIDTH;
    // Search for optimal split size by trying powers of 2 from conservative start to max_dim
    for (uint32_t split_size = start_split_size; split_size <= max_dim; split_size *= 2) {
        // Calculate work distribution for this split size
        const uint32_t rem = width % split_size;                      // Remainder after even division
        const uint32_t num_cores = (width / split_size) + (rem > 0);  // Cores needed (extra for remainder)

        // Calculate memory costs for this configuration:
        // Gather cost: Memory for collecting results from all cores (includes both value and index tiles)
        // Factor of 2 accounts for intermediate storage during gather phase
        const uint32_t memory_cost_gather = 2 * num_cores * (value_tile_size + index_tile_size);

        // Local cost: Memory each core needs for its portion of the work
        // Proportional to split_size converted to tiles
        const uint32_t memory_cost_local =
            (split_size / tt::constants::TILE_WIDTH) * (value_tile_size + index_tile_size);

        // Extract core grid dimensions from the available range
        const uint32_t max_x = core_range.end_coord.x - core_range.start_coord.x;
        const uint32_t max_y = core_range.end_coord.y - core_range.start_coord.y - 1;
        const uint32_t max_cores_available = max_x * max_y;
        // Quick check: skip this configuration if it needs more cores than available
        if (num_cores > max_cores_available) {
            continue;
        }

        // Find contiguous core arrangement that matches the required number of cores
        // Hardware performs better with contiguous rectangular core grids
        bool contiguous_cores_available = false;
        uint32_t selected_x = 0;
        uint32_t selected_y = 0;

        // Search from largest dimensions down to find optimal core grid shape
        // Prefer arrangements that maximize spatial locality
        for (uint32_t y = max_y; y > 0; y--) {
            for (uint32_t x = max_x; x > 0; x--) {
                if (x * y == num_cores) {
                    selected_x = x;
                    selected_y = y;
                    contiguous_cores_available = true;
                    break;  // Take the first (largest) valid arrangement found
                }
            }
        }
        // Comprehensive validation: check all requirements for a valid configuration
        if (num_cores <= max_cores &&                                                        // Core count feasible
            memory_cost_gather + (memory_cost_local * num_cores) < (l1_size * num_cores) &&  // Memory fits
            num_cores > 1 &&                                                                 // Multi-core beneficial
            split_size >= min_dim &&                                                         // Hardware minimum met
            contiguous_cores_available &&                                                    // Can arrange cores
            rem == 0) {  // Perfect division (no remainder)

            // Create configuration with all the calculated parameters
            TopKCoreConfig config{};
            config.num_cores = static_cast<uint16_t>(num_cores);
            config.split_size = static_cast<uint16_t>(split_size);
            config.rem = static_cast<uint16_t>(rem);

            // Calculate final input size after parallel processing:
            // Each core produces top-K results, so final size is num_cores * max(K, TILE_WIDTH)
            // TILE_WIDTH minimum ensures proper tile alignment
            config.final_input_size =
                static_cast<uint16_t>(num_cores * std::max(k, static_cast<uint32_t>(tt::constants::TILE_WIDTH)));

            config.selected_x = static_cast<uint16_t>(selected_x);
            config.selected_y = static_cast<uint16_t>(selected_y);

            // Return the first valid configuration found (greedy approach)
            return std::make_optional(config);
        }
    }
    // No valid configuration found after trying all split sizes
    return std::nullopt;
}

/**
 * @brief Verifies if multi-core TopK execution is feasible
 *
 * This is a convenience function that wraps find_topk_core_config to provide
 * a simple boolean answer: can TopK be executed efficiently on multiple cores
 * given the current constraints?
 *
 * @param width Total width of the dimension being processed
 * @param min_dim Minimum allowed split size
 * @param max_dim Maximum allowed split size
 * @param k Number of top elements to find
 * @param core_range Available core grid
 * @param l1_size L1 cache size per core
 * @param value_tile_size Memory size of value tiles
 * @param index_tile_size Memory size of index tiles
 * @return true if multi-core execution is feasible, false otherwise
 */
bool verify_multi_core_cost(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size) {
    // Attempt to find a valid configuration
    const auto config =
        find_topk_core_config(width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size);
    return config.has_value();
}

/**
 * @brief Verifies if single-core TopK execution fits within memory constraints
 *
 * Analyzes the memory requirements for executing TopK on a single core by calculating
 * the total L1 memory needed for all intermediate buffers (circular buffers) used
 * during the TopK algorithm execution.
 *
 * Memory components analyzed:
 * - Input circular buffers: For streaming input data
 * - Transpose circular buffers: For data reorganization
 * - Result preparation buffers: For intermediate TopK results
 * - Output circular buffers: For final results
 *
 * Each buffer stores both values and indices, and the total memory requirement
 * must fit within the L1 cache size of a single core.
 *
 * @param input_tensor Input tensor to process (used for data type and device info)
 * @param k Number of top elements to find
 * @param uint16_output Whether indices should be 16-bit (vs 32-bit)
 * @return true if single-core execution fits in L1 memory, false otherwise
 */
bool verify_single_core_cost(const ttnn::Tensor& input_tensor, uint32_t k, bool uint16_output) {
    // Circular buffer configuration constants
    constexpr uint32_t num_cb_unit = 2;                // Base unit for buffer sizing
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;  // Input buffer size multiplier

    // Calculate number of tiles needed to store K elements
    const uint32_t Ktiles = tt::div_up(k, tt::constants::TILE_WIDTH);

    // Define circular buffer requirements for different stages:
    constexpr uint32_t input_cb_tile_count = cb_in_units;   // Input data streaming
    constexpr uint32_t transposed_cb_tile_count = 4;        // Data transposition operations
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output storage

    // Get device and determine data formats
    const auto* device = input_tensor.device();
    const tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat index_cb_data_format = uint16_output ? tt::DataFormat::UInt16 : tt::DataFormat::UInt32;

    // Calculate tile sizes for values and indices
    const uint32_t value_tile_size = tt::tile_size(value_cb_data_format);
    const uint32_t index_tile_size = tt::tile_size(index_cb_data_format);

    // Total memory cost: sum of all circular buffers, each storing both values and indices
    const uint32_t memory_cost_local =
        (input_cb_tile_count + transposed_cb_tile_count + result_prep_cb_tile_count + output_cb_tile_count) *
        (value_tile_size + index_tile_size);

    // Verify that total memory requirement fits within single core's L1 cache
    return memory_cost_local < device->l1_size_per_core();
}

}  // namespace ttnn::prim
