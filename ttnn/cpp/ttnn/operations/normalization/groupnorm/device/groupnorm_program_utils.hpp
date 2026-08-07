// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <initializer_list>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

// Non-tile-aligned H*W (#50682): the two-pass path reduces over the tile-padding rows as data, so
// the scaler must divide by the real element count and the compute kernel must subtract K*E[x]^2.
// Shared by all three two-pass factories so the scaler formula has one definition. Kernels re-derive
// `active` from (padded_hw != logical_hw), hence `kernel_logical_hw` reporting padded_hw when off --
// a disagreeing flag would hang the compute kernel on dfb_k.wait_front. Interleaved ships these as
// compile-time args, so H*W=200 and H*W=224 (same padded 224) must not share a cached program; they
// do not, because the default program hash keys on TensorSpec's logical_shape.
struct GroupNormPadCorrection {
    bool active = false;
    uint32_t logical_hw = 0;
    uint32_t padded_hw = 0;
    uint32_t kernel_logical_hw = 0;  // what the kernels are told: logical when active, else padded
    uint32_t k_bits = 0;             // K = padded_hw/logical_hw - 1, as float bits

    // Reduce scaler that divides by the real element count rather than the padded one. The sqrt is
    // because the AVG/REDUCE_SCALAR LLK applies the scaler twice (row then col). Scaling the divisor
    // rather than masking the scaler tile is forced: prepare_reduce_scaler's
    // `valid_reduce_dim_elements_in_tile` is ignored under REDUCE_SCALAR. L/P being a ratio makes
    // this invariant to how H*W splits across cores -- reduce_factor_c still yields L * C_g.
    uint32_t scaler_bits(uint32_t reduce_factor_w) const;
};

GroupNormPadCorrection make_group_norm_pad_correction(uint32_t logical_hw, uint32_t padded_hw, bool use_welford);

// Appends the three single-tile pad-correction CBs when the correction is active: dfb_k (written by
// the writer) plus two scratch tiles. The indices differ per path by what each already occupies.
// Costs 3 tiles per core when active -- 6KB at bfloat16, 12KB at float32.
void append_group_norm_pad_correction_cbs(
    tt::tt_metal::ProgramDescriptor::CBDescriptors& cbs,
    const GroupNormPadCorrection& pad,
    std::array<uint32_t, 3> cb_indices,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::DataFormat data_format,
    uint32_t single_tile_size);

// True when any reconfig-relevant CB format is fp32, so the compute kernel must run its
// reconfig_data_format calls. When all are bf16 those calls are no-ops and the kernel skips them.
bool groupnorm_needs_fp32_reconfig(std::initializer_list<tt::DataFormat> reconfig_formats);

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<tt::tt_metal::CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<tt::tt_metal::CoreCoord>& group,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_first,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_mid,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

}  // namespace ttnn::prim
