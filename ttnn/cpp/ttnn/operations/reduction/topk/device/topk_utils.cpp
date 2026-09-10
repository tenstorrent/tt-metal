// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
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

bool topk_multicore_structurally_eligible(uint32_t reduced_width, uint32_t num_tile_rows, uint32_t k) {
    // Requirement #1: enough width for parallel execution to pay off. The single-core
    // factory parallelizes across tile ROWS, so when the input has at most
    // multi_core_low_ht_max_tile_rows tile rows most of the grid idles and the
    // column-split multi-core path wins from multi_core_low_ht_min_width up
    // (measured ~4x on 32x2048 k=32); wide-and-tall inputs keep the row-parallel
    // single-core path below multi_core_min_width.
    const bool width_gate = (reduced_width >= constants::multi_core_min_width) ||
                            (num_tile_rows <= constants::multi_core_low_ht_max_tile_rows &&
                             reduced_width >= constants::multi_core_low_ht_min_width);
    // Requirement #2: the multi-core bitonic sort network addresses elements with
    // 16-bit indices (the OUTPUT index tensor may still be 16- or 32-bit), and the
    // network requires a power-of-two width.
    const bool is_pow2 = reduced_width != 0 && (reduced_width & (reduced_width - 1)) == 0;
    // Requirement #3: K limit of the local-topk/gather/final-topk pipeline.
    return width_gate && (reduced_width < constants::multi_core_max_width_exclusive) && is_pow2 &&
           (k <= constants::multi_core_max_k);
}

/**
 * @brief Finds optimal core configuration for multi-core TopK execution
 *
 * This function determines the best way to distribute TopK work across multiple cores
 * by analyzing memory constraints, core availability, and workload balance. It evaluates
 * every valid configuration and returns the one with the lowest modeled makespan
 * (see the fitted cost model at the search loop below).
 *
 * Algorithm overview:
 * 1. Start with a conservative split size based on available cores and width
 *    (clamped up to the minimum split so small widths on large grids start valid)
 * 2. Iteratively try larger split sizes (powers of 2) up to max_dim
 * 3. For each split size, calculate required cores and memory costs
 * 4. Verify that configuration fits within available cores and memory
 * 5. Find contiguous core arrangement that matches the requirement
 * 6. Score each valid configuration with the makespan model
 *    (kLocalCostFactor * Wt_local + kFinalCostFactor * Wt_final) and return the minimum
 *
 * Memory cost model:
 * - Gather cost: Data movement between cores (2 * num_cores * tile_sizes)
 * - Local cost: Per-core memory usage (split_size/TILE_WIDTH * tile_sizes)
 * - Total must fit within L1 memory per core
 *
 * Returns std::nullopt (single-core fallback) instead of throwing when the grid or
 * width cannot support the multi-core layout.
 */
namespace {
std::optional<TopKCoreConfig> find_topk_core_config_impl(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size,
    uint32_t tile_width,
    bool first_valid_only) {
    // Grid dimensions (inclusive coordinates). The multi-core layout places the local
    // cores in a (max_x x max_y) rectangle and the final gather core on the row below
    // it, so it needs at least 2 columns and 3 rows; smaller grids (e.g. a single-row
    // sub_core_grids) fall back to single-core via nullopt. Computing the +1 sizes
    // first keeps the arithmetic unsigned-underflow-free.
    const uint32_t grid_x = core_range.end_coord.x - core_range.start_coord.x + 1;
    const uint32_t grid_y = core_range.end_coord.y - core_range.start_coord.y + 1;
    if (grid_x < 2 || grid_y < 3) {
        return std::nullopt;
    }
    const uint32_t max_x = grid_x - 1;
    const uint32_t max_y = grid_y - 2;
    const uint32_t max_cores = max_x * max_y;

    // Calculate conservative starting split size:
    // 1. Divide width by tile width to get number of tiles
    // 2. Divide by largest power-of-two <= max_cores for balanced distribution
    // 3. Convert back to elements by multiplying by tile width
    // This starts the sweep at the smallest split that can utilize most available
    // cores. When the grid has more (power-of-two) cores than the width has tiles
    // the division truncates to zero, so clamp up to the smallest legal split —
    // otherwise widths right at the eligibility floor (e.g. 1024 = 32 tiles on a
    // 96-core grid, lp2 = 64) would produce a zero split instead of a valid config.
    const uint32_t start_split_size = std::max(
        {static_cast<uint32_t>(width / tile_width / largest_power_of_two(max_cores)) * tile_width,
         min_dim,
         tile_width});

    // The transposed intermediate CBs (c_2, c_4, c_6, and c_8 on local cores) all use bf16
    // when the input format is bfp8/bfp4 to avoid shared-exponent precision loss during sort
    // and inter-core transfer.  Use the larger of the two tile sizes for all value-holding
    // CB cost estimates.
    const uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t transposed_tile_size = std::max(value_tile_size, bf16_tile_size);

    // Search all power-of-2 split sizes and keep the one with the best modeled makespan.
    // The first-valid (= smallest split, most cores) choice maximizes the SERIAL final
    // stage: the single final core does O(num_cores * k) gather-merge work while every
    // local core does O(split_size) sort work. Model both sides and minimize
    //   T ~ kLocalCostFactor * Wt_local + kFinalCostFactor * Wt_final
    // Constants fitted on p150a silicon (4 configs across 8192/32768-wide k=64 cells,
    // <0.5% residual): a local tile costs ~3.5x a final tile — locals run full
    // 64-element sorts per tile while the final core runs merge/rebuild pair-ops.
    // Wormhole impact: this selection logic is arch-neutral and changes nothing
    // Wormhole-specific — the low-tile-row eligibility gate applies identically
    // there, and the start-split clamp above only binds at the eligibility floor
    // (a zero split needs lp2(max_cores) > width-in-tiles; Wormhole grids top out
    // at lp2(max_cores) = 32, so W=1024's 32 tiles never truncate to zero there).
    constexpr uint32_t kLocalCostFactor = 7;
    constexpr uint32_t kFinalCostFactor = 2;
    std::optional<TopKCoreConfig> best_config = std::nullopt;
    uint32_t best_score = std::numeric_limits<uint32_t>::max();
    for (uint32_t split_size = start_split_size; split_size <= max_dim; split_size *= 2) {
        const uint32_t rem = width % split_size;                      // Remainder after even division
        const uint32_t num_cores = (width / split_size) + (rem > 0);  // Cores needed (extra for remainder)

        // Per-core L1 footprint mirroring the multi-core factory's CBs: charge the gather/output
        // buffers to a single core (they live on one core), not amortised across all cores.
        // Each local core physically produces ceil(k / tile_width) tiles (the writer strides by
        // Kt tiles), so the gathered width is num_cores * Kt tiles — round K UP to the tile
        // boundary. Warning: a flooring formula (e.g. max(k, tile_width)) undersizes the gather
        // CBs and the final reader's tile count for K values that are not tile multiples.
        const uint32_t Kt = tt::div_up(k, tile_width);
        const uint32_t Wt_final = num_cores * Kt;
        const uint32_t Wt_local = split_size / tile_width;
        const uint32_t shared_cost = 4 * (value_tile_size + index_tile_size) +              // c_0,c_1 input
                                     Wt_final * (transposed_tile_size + index_tile_size) +  // c_4,c_5 gathered
                                     2 * index_tile_size;                                   // c_9 local-index out
        const uint32_t final_core_cost =  // + c_8 value, c_6/c_7 workspace
            shared_cost + 2 * value_tile_size + Wt_final * (transposed_tile_size + index_tile_size);
        const uint32_t local_core_cost =  // + c_2/c_3 transposed, c_8 value
            shared_cost + Wt_local * (transposed_tile_size + index_tile_size) + 2 * transposed_tile_size;
        const uint32_t per_core_cost = std::max(final_core_cost, local_core_cost);

        // Quick check: skip this configuration if it needs more cores than available
        if (num_cores > max_cores) {
            continue;
        }

        // Find a contiguous rectangular core arrangement matching the required core count.
        // Hardware performs better with contiguous rectangular core grids. Scan y upward and
        // take the first factorization that fits: the WIDEST (smallest-y) rectangle. This is
        // the arrangement the split model's cost constants were fitted against on silicon
        // (and what the previous descending double-loop shipped — its inner break only exited
        // the x-loop, so it also kept the smallest-y match despite its comment).
        bool contiguous_cores_available = false;
        uint32_t selected_x = 0;
        uint32_t selected_y = 0;
        for (uint32_t y = 1; y <= max_y; y++) {
            if (num_cores % y != 0) {
                continue;
            }
            const uint32_t x = num_cores / y;
            if (x <= max_x) {
                selected_x = x;
                selected_y = y;
                contiguous_cores_available = true;
                break;
            }
        }

        // Comprehensive validation: check all requirements for a valid configuration.
        const bool valid = num_cores <= max_cores &&      // Core count feasible
                           per_core_cost < l1_size &&     // Memory fits
                           num_cores > 1 &&               // Multi-core beneficial
                           split_size >= min_dim &&       // Hardware minimum met
                           contiguous_cores_available &&  // Can arrange cores
                           rem == 0;                      // Perfect division (no remainder)
        if (!valid) {
            continue;
        }

        // Create configuration with all the calculated parameters
        TopKCoreConfig config{};
        config.num_cores = static_cast<uint16_t>(num_cores);
        config.split_size = static_cast<uint16_t>(split_size);
        config.rem = static_cast<uint16_t>(rem);
        // Final gather width: each core lands Kt = ceil(K / tile_width) tiles, so the
        // final stage input is num_cores * Kt tiles (a power of two whenever num_cores
        // and Kt are, which the final merge's log2(Wt_final) iteration count relies on).
        config.final_input_size = static_cast<uint16_t>(Wt_final * tile_width);
        config.selected_x = static_cast<uint16_t>(selected_x);
        config.selected_y = static_cast<uint16_t>(selected_y);

        // Existence checks (verify_multi_core_cost) do not need the best-scoring
        // config, just whether any valid one exists — return the first and skip the
        // rest of the sweep.
        if (first_valid_only) {
            return config;
        }

        // Only keep a config if it also beats the best modeled makespan so far.
        const uint32_t score = kLocalCostFactor * Wt_local + kFinalCostFactor * Wt_final;
        if (score < best_score) {
            best_score = score;
            best_config = config;
        }
    }
    return best_config;
}
}  // namespace

std::optional<TopKCoreConfig> find_topk_core_config(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size,
    uint32_t tile_width) {
    return find_topk_core_config_impl(
        width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size, tile_width, false);
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
    uint32_t index_tile_size,
    uint32_t tile_width) {
    // Existence is independent of the makespan score: any valid split proves
    // feasibility, so stop the sweep at the first valid config instead of paying
    // the full scored search (this runs on the host dispatch hot path from both
    // select_program_factory and validate_on_program_cache_miss).
    return find_topk_core_config_impl(
               width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size, tile_width, true)
        .has_value();
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
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t Ktiles = tt::div_up(k, tile_width);

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

    // Transposed (c_2) and result-prep (c_4) CBs use bf16 only when input is bfp8/bfp4 (those
    // are upcast to bf16 for the sort). fp32 is kept at full width for the exact fp32 sort, so
    // its compute buffers are fp32-sized — modeling that here lets large-K fp32 be rejected
    // cleanly instead of overflowing L1 at CB-allocation time.
    const uint32_t compute_tile_size =
        (value_cb_data_format == tt::DataFormat::Bfp8_b || value_cb_data_format == tt::DataFormat::Bfp4_b)
            ? tt::tile_size(tt::DataFormat::Float16_b)
            : value_tile_size;

    // Total memory cost: input/output buffers use value_tile_size, intermediate compute
    // buffers (transposed + result_prep) use compute_tile_size (may be larger).
    const uint32_t memory_cost_local =
        (input_cb_tile_count * (value_tile_size + index_tile_size)) +
        ((transposed_cb_tile_count + result_prep_cb_tile_count) * (compute_tile_size + index_tile_size)) +
        (output_cb_tile_count * (value_tile_size + index_tile_size));

    // Verify that total memory requirement fits within single core's L1 cache
    return memory_cost_local < device->l1_size_per_core();
}

bool is_uint32_index_required(const ttnn::Tensor& input_tensor, int8_t dim) {
    const bool dim_fits_uint16 = input_tensor.padded_shape()[dim] <= std::numeric_limits<uint16_t>::max();
    const bool is_fp32 = input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32;
    return !dim_fits_uint16 || is_fp32;
}

}  // namespace ttnn::prim
