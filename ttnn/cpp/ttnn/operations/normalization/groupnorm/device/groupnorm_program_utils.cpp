// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_program_utils.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <algorithm>

#include <tt-metalium/tt_backend_api_types.hpp>  // tt::DataFormat, tt::tile_size
#include <ttnn/tensor/types.hpp>                 // tt::tt_metal::datatype_to_dataformat_converter

namespace ttnn::prim {

uint32_t GroupNormPadCorrection::scaler_bits(uint32_t reduce_factor_w) const {
    const float sc = 1.0f / std::sqrt(
                                static_cast<float>(reduce_factor_w) * static_cast<float>(logical_hw) /
                                static_cast<float>(padded_hw));
    return std::bit_cast<uint32_t>(sc);
}

GroupNormPadCorrection make_group_norm_pad_correction(
    uint32_t logical_hw, uint32_t padded_hw, bool use_welford, uint32_t tile_height) {
    // Welford cannot express this: its kernels transpose H*W into the tile columns and track the
    // sample count in tile units, so the padding rows cannot be excluded. ttnn::group_norm routes
    // non-tile-aligned Welford requests to the two-pass path instead.
    GroupNormPadCorrection pad;
    pad.active = !use_welford && (logical_hw != padded_hw);
    pad.logical_hw = logical_hw;
    pad.padded_hw = padded_hw;
    pad.kernel_logical_hw = pad.active ? logical_hw : padded_hw;
    // padded_hw is logical_hw rounded up to a tile, so active implies a non-zero remainder.
    pad.rows_in_last_tile = pad.active ? (logical_hw % tile_height) : 0;
    return pad;
}

bool groupnorm_needs_fp32_reconfig(std::initializer_list<tt::DataFormat> reconfig_formats) {
    return std::any_of(reconfig_formats.begin(), reconfig_formats.end(), [](tt::DataFormat format) {
        return format != tt::DataFormat::Float16_b;
    });
}

uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt) {
    // Mirrors how the kernel pads num_out_blocks: a leftover remainder becomes extra full-size blocks.
    const uint32_t out_block_h_normal = block_ht / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    if (block_ht % num_out_blocks != 0) {
        const uint32_t residual = block_ht - num_out_blocks * out_block_h_normal;
        num_out_blocks_padded += residual / out_block_h_normal + 1;
    }
    return num_out_blocks_padded * out_block_h_normal * block_wt;
}

uint32_t groupnorm_heuristic_num_out_blocks(uint32_t volume, uint32_t num_virtual_cores) {
    constexpr uint32_t HEURISTIC_BLOCK_SIZE_BASE = 256 * 256;
    constexpr uint32_t MAX_HEURISTIC_NUM_OUT_BLOCKS = 256;
    if (num_virtual_cores == 0) {
        return 1;
    }
    uint32_t heuristic = volume / (HEURISTIC_BLOCK_SIZE_BASE * num_virtual_cores);
    heuristic = heuristic ? heuristic : 1;
    uint32_t num_out_blocks = 1;
    while (num_out_blocks < heuristic && num_out_blocks < MAX_HEURISTIC_NUM_OUT_BLOCKS) {
        num_out_blocks <<= 1;
    }
    return num_out_blocks;
}

bool groupnorm_legacy_rm_input_fits_l1(
    uint32_t Ht,
    uint32_t W,
    uint32_t per_batch_hw,
    uint32_t num_batches,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t num_groups,
    int num_out_blocks_arg,
    uint32_t tile_width,
    uint32_t single_tile_size,
    bool has_gamma,
    bool has_beta,
    bool tilize_in,
    bool untilize_out,
    uint64_t available_l1) {
    // Grid geometry, same formulas as the program factory.
    uint32_t num_virtual_cols = std::min(grid_x, num_groups);
    while (num_virtual_cols > 0 && ((W / num_virtual_cols) % tile_width != 0 || (num_groups % num_virtual_cols) != 0)) {
        num_virtual_cols -= 1;
    }
    if (num_virtual_cols == 0) {
        return false;  // Invalid grid; report "does not fit" (fall back to the composite path)
    }
    const uint32_t num_virtual_rows = (grid_x / num_virtual_cols) * grid_y;
    if (num_virtual_rows == 0 || Ht < num_virtual_rows) {
        return false;
    }
    const uint32_t per_core_Mt = Ht / num_virtual_rows;
    const uint32_t per_core_N = W / num_virtual_cols;
    const uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;
    const uint32_t num_channels_per_group = W / num_groups;

    const uint32_t block_wt = find_max_tile_span(per_core_N, num_channels_per_group).first;
    // Per-core tile height: many batches per core, or one batch split across rows.
    const uint32_t block_ht = (num_batches >= num_virtual_rows) ? (Ht / num_batches) : per_core_Mt;
    if (block_ht == 0) {
        return false;
    }

    // num_out_blocks: -1 means use the factory's power-of-two heuristic.
    uint32_t num_out_blocks;
    if (num_out_blocks_arg < 0) {
        num_out_blocks = groupnorm_heuristic_num_out_blocks(per_batch_hw * W, num_virtual_cols * num_virtual_rows);
    } else {
        num_out_blocks = num_out_blocks_arg == 0 ? 1 : static_cast<uint32_t>(num_out_blocks_arg);
    }
    num_out_blocks = std::min(num_out_blocks, block_ht);

    // CB footprint: seven per-out-block CBs, the resident group, and an allowance for the small ones.
    const uint64_t in0_block_tiles = static_cast<uint64_t>(block_ht / num_out_blocks) * block_wt;
    const uint64_t per_out_block_cb = in0_block_tiles * single_tile_size;
    // Only a row-major input allocates the resident group.
    const uint64_t resident_cb =
        tilize_in ? static_cast<uint64_t>(groupnorm_tilized_group_tiles(block_ht, num_out_blocks, block_wt)) *
                        single_tile_size
                  : 0;

    uint64_t est =
        resident_cb + 7 * per_out_block_cb + static_cast<uint64_t>(kGroupnormSmallCbAllowanceTiles) * single_tile_size;
    if (has_gamma) {
        est += static_cast<uint64_t>(per_core_Nt) * single_tile_size;
    }
    if (has_beta) {
        est += static_cast<uint64_t>(per_core_Nt) * single_tile_size;
    }
    // Mask CB; a mask is always allocated. Assumes bf16 tiles, which over-estimates on purpose.
    est += static_cast<uint64_t>(block_wt) * single_tile_size * 2;
    if (untilize_out) {
        est += 2 * per_out_block_cb;  // c_30 untilize out + c_20 reread
    }

    return est * 100 <= available_l1 * kGroupnormTilizedL1UsagePercent;
}

bool groupnorm_legacy_rm_prefer_composite_for_perf(
    uint32_t num_cores, uint32_t num_virtual_rows, uint32_t num_batches) {
    const bool imbalanced =
        num_virtual_rows != 0 && num_batches >= num_virtual_rows && (num_batches % num_virtual_rows) != 0;
    return num_cores <= kGroupnormLegacyRmMinCoresForOnChip || imbalanced;
}

int get_max_subblock(uint32_t n, uint32_t max_subblock_w) {
    if (n <= max_subblock_w) {
        return n;
    }

    for (int quotient = max_subblock_w; quotient > 1; --quotient) {
        if (n % quotient == 0) {
            return quotient;
        }
    }
    return 1;
}

bool is_rectangle_grid(const std::vector<tt::tt_metal::CoreCoord>& core_coords) {
    if (core_coords.empty()) {
        return true;
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const auto& coord : core_coords) {
        min_x = std::min(min_x, static_cast<int>(coord.x));
        max_x = std::max(max_x, static_cast<int>(coord.x));
        min_y = std::min(min_y, static_cast<int>(coord.y));
        max_y = std::max(max_y, static_cast<int>(coord.y));
    }

    return ((max_x - min_x + 1) * (max_y - min_y + 1)) == static_cast<int>(core_coords.size());
}

void split_and_form_rectangle_grids(
    std::vector<tt::tt_metal::CoreCoord>& group,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_first,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_mid,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_last) {
    size_t remove_front = 0;
    size_t remove_back = 0;
    size_t min_total_removal = group.size();

    for (size_t front = 0; front <= group.size(); ++front) {
        for (size_t back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<tt::tt_metal::CoreCoord>(group.begin() + front, group.end() - back))) {
                size_t total_removal = front + back;
                if (total_removal < min_total_removal) {
                    min_total_removal = total_removal;
                    remove_front = front;
                    remove_back = back;
                }
            }
        }
    }

    // Pop and push the front outliers
    for (size_t i = 0; i < remove_front; ++i) {
        mcast_group_first.push_back(mcast_group_mid.front());
        mcast_group_mid.erase(mcast_group_mid.begin());
    }

    // Pop and push the back outliers
    for (size_t i = 0; i < remove_back; ++i) {
        mcast_group_last.push_back(mcast_group_mid.back());
        mcast_group_mid.pop_back();
    }
}

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width) {
    uint32_t current_position = 0;
    uint32_t max_tile_span = 0;
    uint32_t num_groups_before_start_again_at_tile_beginning = static_cast<uint32_t>(-1);
    bool calc_num_groups_before_start_again_at_tile_beginning = true;

    while (current_position < W) {
        uint32_t group_end = current_position + group_size;
        uint32_t start_tile = current_position / tile_width;
        uint32_t end_tile = (group_end - 1) / tile_width;
        uint32_t current_tile_span = end_tile - start_tile + 1;

        max_tile_span = std::max(max_tile_span, current_tile_span);

        current_position = group_end;

        if (current_position % tile_width == 0 && calc_num_groups_before_start_again_at_tile_beginning) {
            num_groups_before_start_again_at_tile_beginning = current_position / group_size;
            calc_num_groups_before_start_again_at_tile_beginning = false;
        }
    }

    return {max_tile_span, num_groups_before_start_again_at_tile_beginning};
}

uint32_t GroupNormShardedStaticCbSizes::total(const GroupNormShardedCbFlags& flags) const {
    uint32_t t = 0;
    t += in_CB_size;  // c_1 tilized input
    // The no-negative-mask path keeps a second full-shard copy for untilize-out (c_30); the
    // negative-mask path accumulates in place into c_1 and replaces c_30 with the (much
    // smaller) negative-mask CB (c_14). This one line is the whole L1 trade-off.
    t += flags.with_negative_mask ? in_negative_mask_CB_size : (flags.untilize_out ? in_CB_size : 0u);
    t += in2_CB_size;  // c_2 scaler
    t += in3_CB_size;  // c_3 eps
    if (!flags.use_welford) {
        t += in2_CB_size;  // c_4 scaler-c
    }
    if (flags.has_gamma) {
        t += in5_CB_size;  // c_5
    }
    if (flags.has_beta) {
        t += in6_CB_size;  // c_6
    }
    // c_7 is unconditional: the factory allocates it whether the writer NOC-reads a
    // caller-supplied mask or synthesizes one in L1, and one of those is always true.
    t += in_mask_CB_size;
    if (flags.reader_repack_output) {
        t += repack_CB_size;  // c_11/c_12
    }
    t += x_CB_size;           // c_13
    t += ex_partial_CB_size;  // c_8
    if (!flags.use_welford) {
        t += single_tile_size;  // c_10 ex_external
    }
    t += ex_global_CB_size;  // c_9/c_15
    t += ex2pe_CB_size;      // c_17
    t += scalar_tile_size;   // c_26 ones
    if (flags.pad_correction_active) {
        // c_18 rowvalid + c_19 composed mask, created inline by the sharded factory (the
        // append_group_norm_pad_correction_cbs helper this once referred to no longer exists).
        t += rowvalid_CB_size + composed_mask_CB_size;
    }
    return t;
}

GroupNormShardedStaticCbSizes compute_sharded_gn_static_cb_sizes(
    const ttnn::Tensor& input,
    tt::tt_metal::DataType im_data_format,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    std::optional<tt::tt_metal::DataType> input_mask_dtype,
    std::optional<tt::tt_metal::DataType> negative_mask_dtype,
    bool use_welford,
    uint32_t num_groups) {
    using tt::tt_metal::datatype_to_dataformat_converter;

    // Data formats, mirroring groupnorm_sharded_program_factory.cpp. Note beta overrides gamma
    // when both are present -- they share one CB format there, so they must here too.
    const tt::DataFormat in_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(im_data_format);
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    if (gamma_dtype.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(gamma_dtype.value());
    }
    if (beta_dtype.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(beta_dtype.value());
    }
    // Absent mask tensors mean the writer synthesizes them, which it does in bfloat16.
    const tt::DataFormat in_mask_cb_data_format = input_mask_dtype.has_value()
                                                      ? datatype_to_dataformat_converter(input_mask_dtype.value())
                                                      : tt::DataFormat::Float16_b;
    const tt::DataFormat in_negative_mask_cb_data_format =
        negative_mask_dtype.has_value() ? datatype_to_dataformat_converter(negative_mask_dtype.value())
                                        : tt::DataFormat::Float16_b;

    const uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    const uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);
    const uint32_t in_negative_mask_single_tile_size = tt::tile_size(in_negative_mask_cb_data_format);

    // Geometry, again mirroring the factory. Only the derivations the CB sizes depend on.
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const auto& shard_spec = input.shard_spec().value();
    const uint32_t per_core_M = shard_spec.shape[0];
    const uint32_t per_core_N = shard_spec.shape[1];
    const uint32_t per_core_Mt = per_core_M / tile_height;
    const uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;

    const auto& padded_shape = input.padded_shape();
    const uint32_t num_batches = padded_shape[0];
    const uint32_t H = padded_shape[2] * num_batches;
    const uint32_t W = padded_shape[3];
    const uint32_t group_size = W / num_groups;
    const uint32_t block_wt = find_max_tile_span(per_core_N, group_size).first;

    const uint32_t num_shards_r = H / per_core_M;
    const uint32_t num_shards_c = W / per_core_N;
    const uint32_t num_batches_per_core = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    const uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;
    const uint32_t block_ht = per_core_Mt / num_batches_per_core;

    const uint32_t in0_block_tiles = per_core_Nt * per_core_Mt;
    const uint32_t interm_block_tiles = block_ht * block_wt;

    GroupNormShardedStaticCbSizes sizes;
    sizes.in_CB_size = in0_block_tiles * in_single_tile_size;
    // cb_xmm (c_2) double buffer. After the Welford mask-multiply reorder
    // (`((x - mu) * rsqrt) * mask`), only one tile is live in cb_xmm at a time, so the
    // Welford allocation is 2 rather than 3.
    // Scalar CBs are written as bf16 bit patterns, so they stay bf16 even on the legacy fp32
    // path where cb_data_format is Float32. Welford repurposes c_2 as the fp32 cb_xmm
    // intermediate; legacy uses it as the bf16 scaler. Mirrors the factory.
    const uint32_t scalar_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t in2_single_tile_size = use_welford ? single_tile_size : scalar_tile_size;
    sizes.in2_CB_size = in2_single_tile_size * (use_welford ? 2 : 1);
    sizes.in3_CB_size = scalar_tile_size;
    sizes.in5_CB_size = per_core_Nt * gamma_beta_single_tile_size;
    sizes.in6_CB_size = per_core_Nt * gamma_beta_single_tile_size;
    // Non-Welford: double-buffered single set. Caller masks may carry a row-masked second set
    // under the pad correction, but the sharded writer never streams it into c_7.
    sizes.in_mask_CB_size = block_wt * in_mask_single_tile_size * (use_welford ? num_groups_per_core : 2);
    sizes.in_negative_mask_CB_size = block_wt * in_negative_mask_single_tile_size * 2;
    sizes.repack_CB_size = per_core_Nt * in_single_tile_size * 2;
    sizes.x_CB_size = single_tile_size * (use_welford ? 1 : interm_block_tiles);
    // In welford, mean and var are both stored here, so double the size.
    sizes.ex_partial_CB_size = single_tile_size * (use_welford ? 2 : 1);
    sizes.ex_global_CB_size = sizes.ex_partial_CB_size * (use_welford ? num_groups_per_core : 1);
    sizes.ex2pe_CB_size = use_welford ? single_tile_size * num_groups_per_core : sizes.ex_partial_CB_size;
    sizes.single_tile_size = single_tile_size;
    sizes.scalar_tile_size = scalar_tile_size;
    // Pad-correction CBs (c_18 rowvalid, c_19 composed mask), both bf16. Computed
    // unconditionally; total() only counts them when the correction is active.
    sizes.rowvalid_CB_size = scalar_tile_size;
    sizes.composed_mask_CB_size = block_wt * scalar_tile_size;
    return sizes;
}

}  // namespace ttnn::prim
