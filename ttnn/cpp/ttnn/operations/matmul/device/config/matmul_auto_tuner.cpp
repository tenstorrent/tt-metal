// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_auto_tuner.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace ttnn::operations::matmul::auto_tune {

namespace {

// Fast-path first: within each volume tier, (1, N) and (N, 1) are ordered before
// (h, w) shapes where both dimensions exceed 1. Matches the matmul_block helper's
// pack fast path (h == 1 makes row-major == subblock-major with zero overhead).
constexpr std::array<std::pair<uint32_t, uint32_t>, 20> kFastPathFirst = {{
    // volume 8
    {1, 8},
    {8, 1},
    {2, 4},
    {4, 2},
    // volume 7
    {1, 7},
    {7, 1},
    // volume 6
    {1, 6},
    {6, 1},
    {2, 3},
    {3, 2},
    // volume 5
    {1, 5},
    {5, 1},
    // volume 4
    {1, 4},
    {4, 1},
    {2, 2},
    // volume 3
    {1, 3},
    {3, 1},
    // volume 2
    {1, 2},
    {2, 1},
    // volume 1
    {1, 1},
}};

// Legacy ordering from SUBBLOCK_HW_CHOICES in matmul_program_config.cpp. Kept
// verbatim so auto-config regression tests (run at prefer_fast_path=false) see
// byte-identical subblock picks relative to the pre-refactor tuner.
constexpr std::array<std::pair<uint32_t, uint32_t>, 20> kLegacyOrder = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

}  // namespace

uint32_t dst_capacity_from_flags(bool fp32_dest_acc_en, bool dst_full_sync_en) {
    // Mirrors ttnn::get_dest_reg_count for the standard 32x32 tile shape.
    // Half-sync doubles the accessible DST tile count; fp32 accumulation halves it.
    uint32_t base = dst_full_sync_en ? 8u : 16u;
    if (fp32_dest_acc_en) {
        base /= 2u;
    }
    return base;
}

std::vector<std::pair<uint32_t, uint32_t>> enumerate_subblock_options(
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fp32_dest_acc_en,
    bool dst_full_sync_en,
    bool require_legacy_writer) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (per_core_M == 0 || per_core_N == 0) {
        return out;
    }
    const uint32_t cap = dst_capacity_from_flags(fp32_dest_acc_en, dst_full_sync_en);

    // (h, w, sort_key tuple {-volume, fast_path_flag, |h-w|}).
    std::vector<std::tuple<uint32_t, uint32_t, std::tuple<int32_t, int32_t, int32_t>>> candidates;
    for (uint32_t h = 1; h <= cap; ++h) {
        if (per_core_M % h != 0) {
            continue;
        }
        for (uint32_t w = 1; w <= cap; ++w) {
            if (per_core_N % w != 0) {
                continue;
            }
            if (h * w > cap) {
                continue;
            }
            if (require_legacy_writer && !(h == 1 || w == per_core_N)) {
                continue;
            }
            const int32_t volume = static_cast<int32_t>(h * w);
            const int32_t fast_path = (h == 1 || w == 1) ? 0 : 1;
            const int32_t hw_diff = static_cast<int32_t>(h > w ? h - w : w - h);
            candidates.emplace_back(h, w, std::make_tuple(-volume, fast_path, hw_diff));
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return std::get<2>(a) < std::get<2>(b);
    });
    out.reserve(candidates.size());
    for (const auto& [h, w, _key] : candidates) {
        out.emplace_back(h, w);
    }
    return out;
}

std::pair<uint32_t, uint32_t> largest_subblock_from_flags(
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fp32_dest_acc_en,
    bool dst_full_sync_en,
    bool require_legacy_writer) {
    const auto options =
        enumerate_subblock_options(per_core_M, per_core_N, fp32_dest_acc_en, dst_full_sync_en, require_legacy_writer);
    return options.empty() ? std::pair<uint32_t, uint32_t>{1u, 1u} : options.front();
}

L1FootprintEstimate estimate_l1_footprint(const L1EstimateInputs& inputs) {
    const uint64_t pm = inputs.per_core_M;
    const uint64_t pn = inputs.per_core_N;
    const uint64_t ibw = inputs.in0_block_w;
    const uint64_t interm_bytes = inputs.interm_tile_bytes != 0 ? inputs.interm_tile_bytes : inputs.out_tile_bytes;

    L1FootprintEstimate fp;
    fp.out_buf_bytes = pm * pn * inputs.out_tile_bytes;
    fp.interm_buf_bytes = (inputs.fuse_bias || inputs.tile_pack_row_major) ? pm * pn * interm_bytes : 0;
    fp.in0_buf_bytes = static_cast<uint64_t>(inputs.num_buffered_blocks) * pm * ibw * inputs.in0_tile_bytes;
    fp.in1_buf_bytes = static_cast<uint64_t>(inputs.num_buffered_blocks) * pn * ibw * inputs.in1_tile_bytes;
    fp.estimated_bytes = fp.out_buf_bytes + fp.interm_buf_bytes + fp.in0_buf_bytes + fp.in1_buf_bytes;
    fp.fits_wh = fp.estimated_bytes <= L1_BUDGET_BYTES_WORMHOLE;
    fp.fits_bh = fp.estimated_bytes <= L1_BUDGET_BYTES_BLACKHOLE;
    fp.headroom_wh = static_cast<int64_t>(L1_BUDGET_BYTES_WORMHOLE) - static_cast<int64_t>(fp.estimated_bytes);
    fp.headroom_bh = static_cast<int64_t>(L1_BUDGET_BYTES_BLACKHOLE) - static_cast<int64_t>(fp.estimated_bytes);
    return fp;
}

SubblockChoice determine_largest_subblock(const SubblockTuneInputs& inputs) {
    const uint32_t dst_capacity = ttnn::get_dest_reg_count(inputs.compute_kernel_config, inputs.tile_shape);
    const uint32_t max_h = inputs.max_subblock_h.value_or(UINT32_MAX);
    const uint32_t max_w = inputs.max_subblock_w.value_or(UINT32_MAX);

    const auto& table = inputs.prefer_fast_path ? kFastPathFirst : kLegacyOrder;

    for (const auto& [h, w] : table) {
        if (h * w > dst_capacity) {
            continue;
        }
        if (h > max_h || w > max_w) {
            continue;
        }
        if (inputs.per_core_M == 0 || inputs.per_core_N == 0) {
            continue;
        }
        if (inputs.per_core_M % h != 0 || inputs.per_core_N % w != 0) {
            continue;
        }
        if (inputs.subblock_w_eq_per_core_n_required) {
            if (w != inputs.per_core_N && h != 1) {
                continue;
            }
        }
        if (inputs.subblock_h_eq_per_core_m_required) {
            if (h != inputs.per_core_M && w != 1) {
                continue;
            }
        }
        return {h, w};
    }
    return {1, 1};
}

uint32_t determine_largest_in0_block_w(const InBlockWTuneInputs& inputs) {
    if (inputs.Kt == 0 || inputs.max_in0_block_w == 0) {
        return 1;
    }
    // Fixed L1 footprint — independent of in0_block_w.
    const uint32_t interm_footprint =
        inputs.fuse_bias ? (inputs.per_core_M * inputs.per_core_N * inputs.interm_single_tile_size) : 0;
    const uint32_t fixed_footprint =
        (inputs.per_core_M * inputs.per_core_N * inputs.out_single_tile_size) + interm_footprint;

    if (fixed_footprint >= inputs.l1_budget_bytes) {
        return 1;  // Budget already consumed by output/interm alone; be safe.
    }
    const uint32_t remaining = inputs.l1_budget_bytes - fixed_footprint;

    // Per-unit-of-in0_block_w footprint: double-buffered in0 + in1 block tiles.
    const uint32_t per_ibw_footprint = inputs.num_buffered_blocks * ((inputs.per_core_M * inputs.in0_single_tile_size) +
                                                                     (inputs.per_core_N * inputs.in1_single_tile_size));
    if (per_ibw_footprint == 0) {
        return 1;
    }

    const uint32_t l1_capped = remaining / per_ibw_footprint;
    uint32_t capped = std::min({l1_capped, inputs.max_in0_block_w, inputs.Kt});
    if (capped == 0) {
        return 1;
    }

    // Find largest w <= capped that evenly divides Kt.
    for (uint32_t w = capped; w >= 1; --w) {
        if (inputs.Kt % w == 0) {
            return w;
        }
    }
    return 1;
}

}  // namespace ttnn::operations::matmul::auto_tune
