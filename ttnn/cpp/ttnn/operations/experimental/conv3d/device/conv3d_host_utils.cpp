// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_host_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::prim {

namespace {

uint32_t host_utils_largest_divisor_up_to(uint32_t n, uint32_t cap) {
    for (uint32_t d = std::min(n, cap); d >= 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

uint32_t dtype_size_bytes(tt::tt_metal::DataType dtype) {
    switch (dtype) {
        case tt::tt_metal::DataType::FLOAT32: return 4;
        case tt::tt_metal::DataType::BFLOAT16: return 2;
        default: return 0;
    }
}

struct CoreBlockExtents {
    // Output-block extents per dimension for one active core.
    std::vector<uint32_t> t_extents;
    std::vector<uint32_t> h_extents;
    std::vector<uint32_t> w_extents;
};

// Reproduce factory parallelization to enumerate per-core block coverage.
// Returns a list of CoreBlockExtents for each active core (one entry per
// (c_in, c_out, t_idx, h_idx, w_idx) combination that has work).
std::vector<CoreBlockExtents> enumerate_active_core_extents(
    uint32_t T_out,
    uint32_t H_out,
    uint32_t W_out,
    const Conv3dConfig& config,
    uint32_t C_in,
    uint32_t C_in_block,
    uint32_t C_out_num_blocks,
    uint32_t num_cores) {
    const uint32_t T_blocks = tt::div_up(T_out, config.T_out_block);
    const uint32_t H_blocks = tt::div_up(H_out, config.H_out_block);
    const uint32_t W_blocks = tt::div_up(W_out, config.W_out_block);

    const uint32_t C_in_num_blocks = tt::div_up(C_in, C_in_block);
    const uint32_t c_in_pf = std::min(C_in_num_blocks, num_cores);
    const uint32_t cores_per_output = std::max(1u, num_cores / c_in_pf);
    const uint32_t c_out_pf = std::min(C_out_num_blocks, cores_per_output);
    uint32_t rem = cores_per_output / c_out_pf;
    const uint32_t t_pf = std::min(T_blocks, rem);
    rem = rem / t_pf;
    const uint32_t h_pf = std::min(H_blocks, rem);
    rem = rem / h_pf;
    const uint32_t w_pf = std::min(W_blocks, rem);

    const uint32_t t_per_core = tt::div_up(T_blocks, t_pf);
    const uint32_t h_per_core = tt::div_up(H_blocks, h_pf);
    const uint32_t w_per_core = tt::div_up(W_blocks, w_pf);

    std::vector<CoreBlockExtents> result;
    result.reserve(static_cast<size_t>(t_pf) * h_pf * w_pf);
    for (uint32_t t_idx = 0; t_idx < t_pf; t_idx++) {
        const uint32_t t_b_start = t_idx * t_per_core;
        const uint32_t t_b_end = std::min(t_b_start + t_per_core, T_blocks);
        if (t_b_end <= t_b_start) {
            continue;
        }
        for (uint32_t h_idx = 0; h_idx < h_pf; h_idx++) {
            const uint32_t h_b_start = h_idx * h_per_core;
            const uint32_t h_b_end = std::min(h_b_start + h_per_core, H_blocks);
            if (h_b_end <= h_b_start) {
                continue;
            }
            for (uint32_t w_idx = 0; w_idx < w_pf; w_idx++) {
                const uint32_t w_b_start = w_idx * w_per_core;
                const uint32_t w_b_end = std::min(w_b_start + w_per_core, W_blocks);
                if (w_b_end <= w_b_start) {
                    continue;
                }
                CoreBlockExtents ex;
                for (uint32_t tb = t_b_start; tb < t_b_end; tb++) {
                    const uint32_t out_start = tb * config.T_out_block;
                    const uint32_t out_end = std::min((tb + 1) * config.T_out_block, T_out);
                    ex.t_extents.push_back(out_end - out_start);
                }
                for (uint32_t hb = h_b_start; hb < h_b_end; hb++) {
                    const uint32_t out_start = hb * config.H_out_block;
                    const uint32_t out_end = std::min((hb + 1) * config.H_out_block, H_out);
                    ex.h_extents.push_back(out_end - out_start);
                }
                for (uint32_t wb = w_b_start; wb < w_b_end; wb++) {
                    const uint32_t out_start = wb * config.W_out_block;
                    const uint32_t out_end = std::min((wb + 1) * config.W_out_block, W_out);
                    ex.w_extents.push_back(out_end - out_start);
                }
                result.push_back(std::move(ex));
            }
        }
    }
    return result;
}

// Receptive-field extent in input for an output extent of `n_out` along an axis
// with given stride and kernel.
inline uint64_t shard_extent(uint32_t n_out, uint32_t stride, uint32_t k) {
    return static_cast<uint64_t>(n_out - 1) * stride + k;
}

// Per-core activation byte estimates for each candidate execution policy.
struct ScoreResult {
    uint64_t prefetch_only_max = 0;
    uint64_t w_slide_max = 0;
    uint64_t h_slide_max = 0;
    uint64_t prefetch_only_total = 0;
    uint64_t w_slide_total = 0;
    uint64_t h_slide_total = 0;
    bool any_w_transition = false;
    bool any_h_transition = false;
};

ScoreResult score_policies(
    const std::vector<CoreBlockExtents>& cores,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& stride,
    uint32_t C_in_block_bytes) {
    const uint32_t kT = kernel_size[0];
    const uint32_t kH = kernel_size[1];
    const uint32_t kW = kernel_size[2];
    const uint32_t sT = stride[0];
    const uint32_t sH = stride[1];
    const uint32_t sW = stride[2];
    const uint32_t overlap_h = (kH > sH) ? (kH - sH) : 0;
    const uint32_t overlap_w = (kW > sW) ? (kW - sW) : 0;

    ScoreResult res;
    for (const auto& core : cores) {
        if (core.t_extents.empty() || core.h_extents.empty() || core.w_extents.empty()) {
            continue;
        }
        const bool has_w_transition = core.w_extents.size() > 1;
        const bool has_h_transition = core.h_extents.size() > 1;
        if (has_w_transition) {
            res.any_w_transition = true;
        }
        if (has_h_transition) {
            res.any_h_transition = true;
        }

        uint64_t prefetch_only = 0;
        uint64_t w_slide = 0;
        uint64_t h_slide = 0;
        for (uint32_t t_extent : core.t_extents) {
            const uint64_t Tcur = shard_extent(t_extent, sT, kT);
            for (uint32_t h_extent : core.h_extents) {
                const uint64_t Hcur = shard_extent(h_extent, sH, kH);
                for (uint32_t w_extent : core.w_extents) {
                    const uint64_t Wcur = shard_extent(w_extent, sW, kW);
                    prefetch_only += Tcur * Hcur * Wcur;
                }
                // W-slide: first w block full, rest only refill new columns.
                bool first = true;
                for (uint32_t w_extent : core.w_extents) {
                    const uint64_t Wcur = shard_extent(w_extent, sW, kW);
                    if (first) {
                        w_slide += Tcur * Hcur * Wcur;
                        first = false;
                    } else {
                        const uint64_t new_cols = (Wcur > overlap_w) ? (Wcur - overlap_w) : 0;
                        w_slide += Tcur * Hcur * new_cols;
                    }
                }
            }
            // H-slide: outer t,w; inner h. First h block full, rest only refill new rows.
            for (uint32_t w_extent : core.w_extents) {
                const uint64_t Wcur = shard_extent(w_extent, sW, kW);
                bool first = true;
                for (uint32_t h_extent : core.h_extents) {
                    const uint64_t Hcur = shard_extent(h_extent, sH, kH);
                    if (first) {
                        h_slide += Tcur * Hcur * Wcur;
                        first = false;
                    } else {
                        const uint64_t new_rows = (Hcur > overlap_h) ? (Hcur - overlap_h) : 0;
                        h_slide += Tcur * new_rows * Wcur;
                    }
                }
            }
        }
        prefetch_only *= C_in_block_bytes;
        w_slide *= C_in_block_bytes;
        h_slide *= C_in_block_bytes;
        res.prefetch_only_max = std::max(res.prefetch_only_max, prefetch_only);
        res.w_slide_max = std::max(res.w_slide_max, w_slide);
        res.h_slide_max = std::max(res.h_slide_max, h_slide);
        res.prefetch_only_total += prefetch_only;
        res.w_slide_total += w_slide;
        res.h_slide_total += h_slide;
    }
    return res;
}

}  // namespace

Conv3dExecutionPolicy resolve_conv3d_execution_policy(
    const Conv3dParams& params, const ttnn::Shape& input_shape, tt::tt_metal::DataType input_dtype, bool has_bias) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& config = params.config;
    const uint32_t T_in = input_shape[1];
    const uint32_t H_in = input_shape[2];
    const uint32_t W_in = input_shape[3];
    const uint32_t C_in = input_shape[4];

    auto [T_out, H_out, W_out] = detail::compute_output_dims(
        T_in, H_in, W_in, params.padding, params.stride, params.kernel_size, params.dilation);

    const uint32_t kT = params.kernel_size[0];
    const uint32_t kH = params.kernel_size[1];
    const uint32_t kW = params.kernel_size[2];

    const uint32_t dtype_bytes = dtype_size_bytes(input_dtype);
    const uint32_t C_out = params.output_channels;
    const uint32_t padded_C_out = round_up(C_out, constants::TILE_WIDTH);
    const uint32_t C_out_block = config.C_out_block > 0 ? config.C_out_block : padded_C_out;
    const uint32_t C_in_block = config.C_in_block > 0 ? config.C_in_block : C_in;

    const uint32_t patch_size = kT * kH * kW * C_in_block;
    const uint32_t padded_patch_size = round_up(patch_size, constants::TILE_WIDTH);
    const uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

    const uint32_t padded_patch_size_bytes = padded_patch_size * dtype_bytes;
    const uint32_t C_in_block_bytes = C_in_block * dtype_bytes;

    const auto data_format = datatype_to_dataformat_converter(input_dtype);
    const uint32_t tile_size = tt::tile_size(data_format);

    const uint32_t C_in_num_blocks = div_up(C_in, C_in_block);
    const uint32_t C_out_num_blocks = div_up(padded_C_out, C_out_block);

    const uint32_t matmul_M_t = div_up(num_patches, constants::TILE_HEIGHT);
    const uint32_t matmul_K_t = div_up(patch_size, constants::TILE_WIDTH);
    const uint32_t matmul_N_t = div_up(C_out_block, constants::TILE_WIDTH);

    auto [_mf, _ma, fp32_dest_acc_en, _pl, _ds] =
        get_compute_kernel_config_args(hal::get_arch(), params.compute_kernel_config);
    (void)_mf;
    (void)_ma;
    (void)_pl;
    (void)_ds;

    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    const bool scale_subblock_h = hal::get_arch() == tt::ARCH::WORMHOLE_B0 && out_subblock_w == matmul_N_t;
    const uint32_t out_subblock_h =
        scale_subblock_h ? host_utils_largest_divisor_up_to(matmul_M_t, dst_size / out_subblock_w) : 1;

    const uint32_t vol2col_rm_pages = (num_patches % constants::TILE_HEIGHT == 0)
                                          ? std::min(num_patches, (uint32_t)constants::TILE_HEIGHT)
                                          : std::min(num_patches, 2 * (uint32_t)constants::TILE_HEIGHT);

    const bool use_fp32_partials = fp32_dest_acc_en && C_in_num_blocks > 1;
    const auto partial_data_format = use_fp32_partials ? tt::DataFormat::Float32 : data_format;
    const uint32_t partial_tile_size = tt::tile_size(partial_data_format);

    uint32_t other_cbs_bytes = (padded_patch_size_bytes * vol2col_rm_pages) +   // vol2col_rm
                               (tile_size * out_subblock_h * matmul_K_t) +      // vol2col_tiled
                               (tile_size * matmul_K_t * matmul_N_t) +          // weight_tiled
                               (partial_tile_size * matmul_M_t * matmul_N_t) +  // matmul_interm
                               (tile_size * matmul_M_t * matmul_N_t);           // matmul_result_rm
    if (C_in_num_blocks > 1) {
        other_cbs_bytes += partial_tile_size * matmul_M_t * matmul_N_t;  // reduction
        other_cbs_bytes += tile_size;                                    // worker_ack
    }
    if (use_fp32_partials) {
        other_cbs_bytes += tile_size;  // zero tile
    }
    if (has_bias) {
        other_cbs_bytes += tile_size * matmul_N_t;  // bias
    }

    constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
    constexpr uint32_t L1_PREFETCH_HARD_CAP = 500 * 1024;
    const uint32_t l1_usable_for_cbs = hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;
    const uint32_t l1_prefetch_max_bytes =
        (other_cbs_bytes < l1_usable_for_cbs) ? std::min(l1_usable_for_cbs - other_cbs_bytes, L1_PREFETCH_HARD_CAP) : 0;

    const bool has_spatial_reuse = (kT > 1 || kH > 1 || kW > 1);
    const bool has_no_dilation = (params.dilation[0] == 1 && params.dilation[1] == 1 && params.dilation[2] == 1);

    Conv3dExecutionPolicy policy;
    if (!has_spatial_reuse || !has_no_dilation) {
        policy.use_l1_prefetch = false;
        policy.slide_axis = Conv3dSlideAxis::None;
        return policy;
    }

    const uint32_t T_shard_max = (config.T_out_block - 1) * params.stride[0] + kT;
    const uint32_t H_shard_max = (config.H_out_block - 1) * params.stride[1] + kH;
    const uint32_t W_shard_max = (config.W_out_block - 1) * params.stride[2] + kW;
    const uint32_t shard_positions_max = T_shard_max * H_shard_max * W_shard_max;
    const uint32_t shard_bytes = shard_positions_max * C_in_block_bytes;

    if (shard_bytes > l1_prefetch_max_bytes) {
        policy.use_l1_prefetch = false;
        policy.slide_axis = Conv3dSlideAxis::None;
        return policy;
    }

    policy.use_l1_prefetch = true;

    // Step 2: pick slide axis.
    const uint32_t num_cores = config.compute_with_storage_grid_size.x * config.compute_with_storage_grid_size.y;
    const auto cores =
        enumerate_active_core_extents(T_out, H_out, W_out, config, C_in, C_in_block, C_out_num_blocks, num_cores);
    const auto scores = score_policies(cores, params.kernel_size, params.stride, C_in_block_bytes);

    const uint32_t overlap_w = (kW > params.stride[2]) ? (kW - params.stride[2]) : 0;
    const uint32_t overlap_h = (kH > params.stride[1]) ? (kH - params.stride[1]) : 0;
    // shift_retained_h_rows / shift_retained_w_columns do an in-place copy with
    // src window [SHARD_MAX - overlap, SHARD_MAX) and dst [0, overlap). Disjoint
    // iff SHARD_MAX >= 2*overlap, otherwise the copy races. Gate the slide axis
    // on this constraint.
    const bool w_shard_safe = (W_shard_max >= 2 * overlap_w) || overlap_w == 0;
    const bool h_shard_safe = (H_shard_max >= 2 * overlap_h) || overlap_h == 0;
    const bool w_eligible = (overlap_w > 0) && scores.any_w_transition && w_shard_safe;
    const bool h_eligible = (overlap_h > 0) && scores.any_h_transition && h_shard_safe;

    // Selection layered for robustness:
    //   1. Default to PrefetchOnly (no slide).
    //   2. Pick W-slide if its DRAM bytes are strictly less than PrefetchOnly.
    //   3. Pick H-slide over W-slide only when H wins on DRAM by ≥20% margin
    //      to absorb unmodeled writer/compute overhead (writer's stride-W_out
    //      output writes, extra L1↔L1 NOC dispatches per H transition).
    enum class Choice { None, W, H };
    Choice chosen = Choice::None;
    if (w_eligible && scores.w_slide_max < scores.prefetch_only_max) {
        chosen = Choice::W;
    }
    if (h_eligible && scores.h_slide_max < scores.prefetch_only_max) {
        const uint64_t baseline = (chosen == Choice::W) ? scores.w_slide_max : scores.prefetch_only_max;
        constexpr uint64_t margin_num = 80;
        constexpr uint64_t margin_den = 100;
        const uint64_t threshold = (baseline * margin_num) / margin_den;
        if (scores.h_slide_max < threshold) {
            chosen = Choice::H;
        }
    }

    switch (chosen) {
        case Choice::None: policy.slide_axis = Conv3dSlideAxis::None; break;
        case Choice::W: policy.slide_axis = Conv3dSlideAxis::W; break;
        case Choice::H: policy.slide_axis = Conv3dSlideAxis::H; break;
    }

    // Debug-only override (must be set BEFORE launch so it lands in the hashed
    // operation attributes — the resolver runs once per launch in conv3d.cpp,
    // and the factory recompute uses the same env at the same point in time).
    if (const char* override_env = std::getenv("TT_METAL_CONV3D_SLIDE_AXIS")) {
        if (std::strcmp(override_env, "none") == 0) {
            policy.slide_axis = Conv3dSlideAxis::None;
        } else if (std::strcmp(override_env, "w") == 0 && (overlap_w > 0) && w_shard_safe) {
            policy.slide_axis = Conv3dSlideAxis::W;
        } else if (std::strcmp(override_env, "h") == 0 && (overlap_h > 0) && h_shard_safe) {
            policy.slide_axis = Conv3dSlideAxis::H;
        }
    }
    return policy;
}

}  // namespace ttnn::experimental::prim
