// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
