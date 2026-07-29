// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

// Non-tile-aligned H*W (#50682): the two-pass path reduces over the tile-padding rows as if they
// were data, so the reduce scaler must divide by the real element count and the compute kernel must
// subtract K*E[x]^2 from the variance. Shared by all three two-pass factories so the scaler formula
// has one definition. Kernels re-derive `active` from (padded_hw != logical_hw), which is why
// `kernel_logical_hw` reports padded_hw when off -- otherwise the kernel flag could disagree with
// the CB allocation and hang the compute kernel on cb_k.wait_front.
struct GroupNormPadCorrection {
    bool active = false;
    uint32_t logical_hw = 0;
    uint32_t padded_hw = 0;
    uint32_t kernel_logical_hw = 0;  // what the kernels are told: logical when active, else padded
    uint32_t k_bits = 0;             // K = padded_hw/logical_hw - 1, as float bits

    // Reduce scaler that divides by the real element count rather than the padded one. The sqrt is
    // because the AVG/REDUCE_SCALAR LLK applies the scaler twice (row then col).
    uint32_t scaler_bits(uint32_t reduce_factor_w) const;
};

GroupNormPadCorrection make_group_norm_pad_correction(uint32_t logical_hw, uint32_t padded_hw, bool use_welford);

// Appends the three single-tile pad-correction CBs when the correction is active: cb_k (written by
// the writer) plus two scratch tiles. The indices differ per path by what each already occupies.
void append_group_norm_pad_correction_cbs(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs,
    const GroupNormPadCorrection& pad,
    std::array<uint32_t, 3> cb_indices,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::DataFormat data_format,
    uint32_t single_tile_size);

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<tt::tt_metal::CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<tt::tt_metal::CoreCoord>& group,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_first,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_mid,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

}  // namespace ttnn::prim
