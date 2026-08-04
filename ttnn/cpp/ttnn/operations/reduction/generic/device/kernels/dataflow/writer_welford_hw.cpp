// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-reduction writer kernel.
//
// Phase 1 (per output): Reads Wt partial (mean, var) tile pairs from
// cb_partial (written by the compute kernel using
// welford_finalize_to_row), combines their equal-sized populations across W,
// applies Bessel's correction, and writes the combined scalar into cb_combined
// for the compute kernel to apply
// sqrtf (if std) and re-pack in the output format. cb_combined is
// normally fp32, but for variance output to bf16 the program
// factory may declare it as bf16 to save SRAM with no precision loss
// since data is packed to bf16 output anyways and there is no math before
// the final pack. combined_is_bf16 compile-time arg selects the path.
//
// Phase 2 (per output): Waits for the compute kernel to pack the
// output tile into cb_out (in the correct output data format), then
// NOC-writes it to DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/numeric/bfloat16.h"
#include "api/tensor/noc_traits.h"
#include <tt-metalium/constants.hpp>

namespace {

constexpr std::uint32_t welford_block_size = tt::constants::TILE_WIDTH;

struct WelfordBlockStats {
    float mean;
    // sum(partial variances) + M2(partial means)
    float variance_sum;
};

constexpr std::uint32_t tree_levels_for(std::uint32_t num_blocks) {
    std::uint32_t levels = 1;
    while (num_blocks > 1) {
        num_blocks >>= 1;
        ++levels;
    }
    return levels;
}

template <std::uint32_t COUNT>
inline WelfordBlockStats finalize_block(
    float base_mean, float mean_delta_sum, float mean_delta_sq_sum, float partial_var_sum) {
    static_assert(COUNT > 0);
    constexpr float inv_count = 1.0f / static_cast<float>(COUNT);
    const float mean_delta = mean_delta_sum * inv_count;
    const float raw_means_m2 = mean_delta_sq_sum - mean_delta_sum * mean_delta;
    const float means_m2 = raw_means_m2 < 0.0f ? 0.0f : raw_means_m2;
    return {.mean = base_mean + mean_delta, .variance_sum = partial_var_sum + means_m2};
}

inline WelfordBlockStats combine_known_counts(
    const WelfordBlockStats& a, const WelfordBlockStats& b, float b_fraction, float cross_weight) {
    const float delta = b.mean - a.mean;
    return {
        .mean = a.mean + delta * b_fraction,
        .variance_sum = a.variance_sum + b.variance_sum + delta * delta * cross_weight};
}

inline void push_full_block(WelfordBlockStats* tree, WelfordBlockStats block, std::uint32_t completed_blocks) {
    std::uint32_t level = 0;
    float half_block_count = static_cast<float>(welford_block_size) * 0.5f;

    // completed_blocks is a binary carry mask: an occupied bit means that level
    // already contains a block of welford_block_size * 2^level partials.
    while ((completed_blocks & 1U) != 0U) {
        block = combine_known_counts(tree[level], block, 0.5f, half_block_count);
        completed_blocks >>= 1;
        ++level;
        half_block_count *= 2.0f;
    }
    tree[level] = block;
}

template <std::uint32_t LEVEL, std::uint32_t NUM_TREE_LEVELS, std::uint32_t NUM_FULL_BLOCKS, std::uint32_t BLOCK_SIZE>
inline void finalize_tree(const WelfordBlockStats* tree, WelfordBlockStats& result) {
    if constexpr (LEVEL < NUM_TREE_LEVELS) {
        constexpr std::uint32_t level_bit = 1U << LEVEL;
        if constexpr ((NUM_FULL_BLOCKS & level_bit) != 0U) {
            constexpr std::uint32_t prior_blocks = NUM_FULL_BLOCKS & (level_bit - 1U);
            if constexpr (prior_blocks == 0) {
                result = tree[LEVEL];
            } else {
                constexpr std::uint32_t a_count = prior_blocks * BLOCK_SIZE;
                constexpr std::uint32_t b_count = level_bit * BLOCK_SIZE;
                constexpr std::uint32_t total_count = a_count + b_count;
                constexpr float inv_total_count = 1.0f / static_cast<float>(total_count);
                constexpr float b_fraction = static_cast<float>(b_count) * inv_total_count;
                constexpr float cross_weight =
                    static_cast<float>(a_count) * static_cast<float>(b_count) * inv_total_count;
                result = combine_known_counts(result, tree[LEVEL], b_fraction, cross_weight);
            }
        }
        finalize_tree<LEVEL + 1, NUM_TREE_LEVELS, NUM_FULL_BLOCKS, BLOCK_SIZE>(tree, result);
    }
}

}  // namespace

void kernel_main() {
    const std::uint32_t dst_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t NC_per_core = get_arg_val<std::uint32_t>(1);
    const std::uint32_t output_tile_start_id = get_arg_val<std::uint32_t>(2);

    constexpr std::uint32_t Wt = get_compile_time_arg_val(0);
    constexpr std::uint32_t W = get_compile_time_arg_val(1);
    constexpr std::uint32_t tile_width = get_compile_time_arg_val(2);
    constexpr std::uint32_t H = get_compile_time_arg_val(3);
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    constexpr std::uint32_t reduce_batch_size = get_compile_time_arg_val(5);
    constexpr bool combined_is_bf16 = get_compile_time_arg_val(6) != 0;
    static_assert(tile_width == welford_block_size);

    constexpr std::uint32_t num_partials = reduce_batch_size * W;
    static_assert(num_partials > 0);
    constexpr std::uint32_t num_full_blocks = num_partials / welford_block_size;
    constexpr std::uint32_t tail_size = num_partials % welford_block_size;
    constexpr std::uint32_t num_tree_levels = tree_levels_for(num_full_blocks);

    constexpr auto cb_partial = tt::CBIndex::c_21;
    // cb_combined: combined scalar tile written by this kernel, read back by
    // compute for repacking into the output data format. Format is Float32 by
    // default; bf16 when combined_is_bf16 is true (variance-to-bf16 path).
    constexpr auto cb_combined = tt::CBIndex::c_22;
    // cb_out: output tile packed by compute in the correct data format.
    constexpr auto cb_out = tt::CBIndex::c_16;

    constexpr auto dst_args = TensorAccessorArgs<7>();

    // welford_finalize_to_row stores 32 per-column values in tile row 0.
    // In tile format, row 0 spans Face 0 (columns 0-15) and Face 1 (columns 16-31).
    // Each face has FACE_W rows × FACE_W columns elements.
    constexpr std::uint32_t FACE_W = tt::constants::FACE_WIDTH;
    constexpr std::uint32_t FACE_ELEMENTS = FACE_W * FACE_W;
    constexpr std::uint32_t last_tile_cols = (W % tile_width == 0) ? tile_width : W % tile_width;

    const std::uint32_t partial_tile_size_bytes = get_tile_size(cb_partial);
    const std::uint32_t out_tile_size_bytes = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer cb_partial_obj(cb_partial);
    CircularBuffer cb_combined_obj(cb_combined);
    CircularBuffer cb_out_obj(cb_out);

    const auto tensor_out = TensorAccessor(dst_args, dst_addr);

    // NC_per_core is the total number of NC slices assigned to this core.
    // Each output element is produced by combining reduce_batch_size
    // consecutive NC slices (each contributing Wt partial tile pairs).
    std::uint32_t num_outputs = NC_per_core / reduce_batch_size;

    for (std::uint32_t out = 0; out < num_outputs; ++out) {
        // --- Phase 1: W-combine all per-column partials into one scalar ---
        // Build stable 32-partial leaves, then merge equal-sized leaves through
        // a binary carry tree. Every tree merge is division-free, and no raw
        // second-moment subtraction spans more than one leaf.
        WelfordBlockStats tree[num_tree_levels];
        std::uint32_t completed_blocks = 0;

        float block_base_mean = 0.0f;
        float block_mean_delta_sum = 0.0f;
        float block_mean_delta_sq_sum = 0.0f;
        float block_partial_var_sum = 0.0f;
        std::uint32_t block_count = 0;

        for (std::uint32_t b = 0; b < reduce_batch_size; ++b) {
            for (std::uint32_t wt = 0; wt < Wt; ++wt) {
                cb_partial_obj.wait_front(2);

                auto means_addr = cb_partial_obj.get_read_ptr();
                auto vars_addr = means_addr + partial_tile_size_bytes;

                // cb_partial is Float32: each element is 4 bytes.
                auto* means_ptr = reinterpret_cast<volatile float*>(means_addr);
                auto* vars_ptr = reinterpret_cast<volatile float*>(vars_addr);

                std::uint32_t num_cols = (wt < Wt - 1) ? tile_width : last_tile_cols;
                for (std::uint32_t c = 0; c < num_cols; ++c) {
                    // In tile row format, columns 0-15 are in Face 0 and
                    // columns 16-31 are in Face 1 (offset by FACE_ELEMENTS).
                    std::uint32_t idx = (c < FACE_W) ? c : (FACE_ELEMENTS + c - FACE_W);
                    const float partial_mean = means_ptr[idx];
                    const float partial_var = vars_ptr[idx];

                    // Every partial summarizes the same H samples. The total population
                    // variance is therefore the average partial variance plus the
                    // population variance of the partial means.
                    if (block_count == 0) {
                        block_base_mean = partial_mean;
                        block_mean_delta_sum = 0.0f;
                        block_mean_delta_sq_sum = 0.0f;
                        block_partial_var_sum = partial_var;
                    } else {
                        const float delta = partial_mean - block_base_mean;
                        block_mean_delta_sum += delta;
                        block_mean_delta_sq_sum += delta * delta;
                        block_partial_var_sum += partial_var;
                    }
                    ++block_count;

                    if (block_count == welford_block_size) {
                        auto block = finalize_block<welford_block_size>(
                            block_base_mean, block_mean_delta_sum, block_mean_delta_sq_sum, block_partial_var_sum);
                        push_full_block(tree, block, completed_blocks);
                        ++completed_blocks;
                        block_count = 0;
                    }
                }

                cb_partial_obj.pop_front(2);
            }
        }

        WelfordBlockStats combined;
        if constexpr (num_full_blocks > 0) {
            finalize_tree<0, num_tree_levels, num_full_blocks, welford_block_size>(tree, combined);
        }

        if constexpr (tail_size > 0) {
            const auto tail = finalize_block<tail_size>(
                block_base_mean, block_mean_delta_sum, block_mean_delta_sq_sum, block_partial_var_sum);
            if constexpr (num_full_blocks == 0) {
                combined = tail;
            } else {
                constexpr std::uint32_t full_count = num_full_blocks * welford_block_size;
                constexpr float inv_num_partials = 1.0f / static_cast<float>(num_partials);
                constexpr float tail_fraction = static_cast<float>(tail_size) * inv_num_partials;
                constexpr float cross_weight =
                    static_cast<float>(full_count) * static_cast<float>(tail_size) * inv_num_partials;
                combined = combine_known_counts(combined, tail, tail_fraction, cross_weight);
            }
        }

        constexpr float inv_num_partials = 1.0f / static_cast<float>(num_partials);
        float final_var;
        if constexpr (correction) {
            constexpr std::uint32_t sample_count = num_partials * H;
            // variance_sum / num_partials is the population variance. Folding the sample
            // count correction into it cancels num_partials from the divisor.
            constexpr float correction_scale = static_cast<float>(H) / static_cast<float>(sample_count - 1);
            final_var = combined.variance_sum * correction_scale;
        } else {
            final_var = combined.variance_sum * inv_num_partials;
        }

        // Write the combined scalar into a tile in cb_combined.  The compute
        // kernel will unpack this and re-pack into cb_out in the correct
        // output data format (using the packer hardware).
        //
        // Only Face 0 row 0 (FACE_W elements) needs zeroing.  The scalar
        // lives at position [0,0]; the remaining FACE_W-1 elements in
        // the same row share a BFP exponent group, so they must be zero
        // to avoid corrupting the scalar's mantissa precision in
        // BFLOAT8_B output.  Other face rows have independent exponents
        // and are never read (the output is a single scalar), so stale
        // L1 contents there are harmless.
        cb_combined_obj.reserve_back(1);
        if constexpr (combined_is_bf16) {
            auto* combined_ptr = reinterpret_cast<std::uint16_t*>(cb_combined_obj.get_write_ptr());
            for (std::uint32_t i = 0; i < FACE_W; ++i) {
                combined_ptr[i] = 0;
            }
            // fp32_to_bf16 applies round-to-nearest-even, matching the packer
            // hardware so the output is bit-identical to a packer-produced bf16.
            combined_ptr[0] = fp32_to_bf16(final_var);
        } else {
            auto* combined_ptr = reinterpret_cast<float*>(cb_combined_obj.get_write_ptr());
            for (std::uint32_t i = 0; i < FACE_W; ++i) {
                combined_ptr[i] = 0.0f;
            }
            combined_ptr[0] = final_var;
        }
        cb_combined_obj.push_back(1);

        // --- Phase 2: NOC-write the output tile (packed by compute) to DRAM ---
        cb_out_obj.wait_front(1);
        std::uint32_t out_tile_id = output_tile_start_id + out;
        noc.async_write(cb_out_obj, tensor_out, out_tile_size_bytes, {}, {.page_id = out_tile_id});
        noc.async_writes_flushed();
        cb_out_obj.pop_front(1);
    }

    noc.async_write_barrier();
}
