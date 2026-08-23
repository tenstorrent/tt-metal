// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <initializer_list>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/tensor/tensor.hpp"  // ttnn::Tensor, tt::tt_metal::DataType

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

// Non-tile-aligned H*W: the reduce scaler must divide by the real element count (`scaler_bits`),
// and the padding rows must be excluded from both accumulation passes. The interleaved kernels do
// that by switching to a row-masked set of input-mask tiles on each batch's final row-tile, of
// which `rows_in_last_tile` are real; the sharded kernels compose that row mask on device from a
// rowvalid tile (c_18) and the column selector. Shared by all three two-pass factories. Kernels
// re-derive `active` from (padded_hw != logical_hw), hence kernel_logical_hw reporting padded_hw
// when off.
struct GroupNormPadCorrection {
    bool active = false;
    uint32_t logical_hw = 0;
    uint32_t padded_hw = 0;
    uint32_t kernel_logical_hw = 0;  // logical when active, else padded
    uint32_t rows_in_last_tile = 0;  // logical_hw % tile_height

    // Reduce scaler that divides by the real element count rather than the padded one. The sqrt is
    // because the AVG/REDUCE_SCALAR LLK applies the scaler twice (row then col). Scaling the divisor
    // rather than masking the scaler tile is forced: prepare_reduce_scaler's
    // `valid_reduce_dim_elements_in_tile` is ignored under REDUCE_SCALAR. L/P being a ratio makes
    // this invariant to how H*W splits across cores -- reduce_factor_c still yields L * C_g.
    uint32_t scaler_bits(uint32_t reduce_factor_w) const;
};

GroupNormPadCorrection make_group_norm_pad_correction(
    uint32_t logical_hw, uint32_t padded_hw, bool use_welford, uint32_t tile_height = 32);

// A batch's padding rows sit in its LAST row-tile, so only the core holding that row-tile applies
// the row mask. `m_index` is virtual_core.y for the interleaved factories, core_index /
// num_shards_c for the sharded one.
inline bool group_norm_core_owns_pad_tile(uint32_t m_index, uint32_t num_cores_per_batch) {
    return (m_index % num_cores_per_batch) == (num_cores_per_batch - 1);
}

// True when any reconfig-relevant CB format is fp32, so the compute kernel must run its
// reconfig_data_format calls. When all are bf16 those calls are no-ops and the kernel skips them.
bool groupnorm_needs_fp32_reconfig(std::initializer_list<tt::DataFormat> reconfig_formats);

constexpr bool groupnorm_uses_sfpu_two_pass(bool use_welford, bool input_requires_tilize) {
    return use_welford && !input_requires_tilize;
}

struct GroupNormInterleavedCbFootprint {
    std::uint64_t output = 0;
    std::uint64_t input_staging = 0;
    std::uint64_t untilize_output = 0;
    std::uint64_t scaler = 0;
    std::uint64_t epsilon = 0;
    std::uint64_t column_scaler = 0;
    std::uint64_t gamma = 0;
    std::uint64_t beta = 0;
    std::uint64_t input_mask = 0;
    std::uint64_t repack = 0;
    std::uint64_t x = 0;
    std::uint64_t xmm = 0;
    std::uint64_t xmm2 = 0;
    std::uint64_t xmm3 = 0;
    std::uint64_t partial_stats = 0;
    std::uint64_t global_stats = 0;
    std::uint64_t normalisation_stats = 0;
    std::uint64_t reciprocals = 0;

    constexpr std::uint64_t total_with_input(std::uint64_t input) const {
        return input + output + input_staging + untilize_output + scaler + epsilon + column_scaler + gamma + beta +
               input_mask + repack + x + xmm + xmm2 + xmm3 + partial_stats + global_stats + normalisation_stats +
               reciprocals;
    }
};

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<tt::tt_metal::CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<tt::tt_metal::CoreCoord>& group,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_first,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_mid,
    std::vector<tt::tt_metal::CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

// Tiles the row-major path keeps resident in c_17 for one per-core group.
uint32_t groupnorm_tilized_group_tiles(uint32_t block_ht, uint32_t num_out_blocks, uint32_t block_wt);

// Auto-select num_out_blocks from tensor volume / virtual core count: next power of two,
// capped at 256. Shared by the program factories and the L1-fit estimate.
// `volume` is H * W * C (padded), `num_virtual_cores` is num_virtual_cols * num_virtual_rows.
uint32_t groupnorm_heuristic_num_out_blocks(uint32_t volume, uint32_t num_virtual_cores);

// Percent of usable L1 we allow the estimate to reach; the margin covers the approximated small CBs.
inline constexpr uint64_t kGroupnormTilizedL1UsagePercent = 95;

// Flat tile budget covering the small 1-tile CBs (eps, ex/ex2 partials, etc.)
inline constexpr uint32_t kGroupnormSmallCbAllowanceTiles = 32;

// At or below this many active cores, prefer composite over fused RM.
inline constexpr uint32_t kGroupnormLegacyRmMinCoresForOnChip = 32;

// Estimates whether a legacy (non-Welford) group_norm fits in L1; if not, group_norm()
// tilizes/untilizes as separate ops. Over-estimates on purpose so "fits" is always safe.
// `tilize_in` adds the resident group; `untilize_out` adds the RM output scratch.
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
    uint64_t available_l1);

// Prefer composite (host tilize + TILE GN) over fused RM for small grids or uneven batch mapping.
// num_cores = num_virtual_cols * num_virtual_rows.
bool groupnorm_legacy_rm_prefer_composite_for_perf(uint32_t num_cores, uint32_t num_virtual_rows, uint32_t num_batches);

// Which of the optional static CBs the sharded factory will emit.
struct GroupNormShardedCbFlags {
    // Selects the negative-mask CB (c_14) in place of the untilize-out copy (c_30) -- the
    // overlap trick the negative mask exists for. True whenever the factory sees either a
    // caller-supplied negative_mask or synthesize_negative_mask.
    bool with_negative_mask = false;
    bool untilize_out = false;
    bool has_gamma = false;
    bool has_beta = false;
    bool reader_repack_output = false;
    bool use_welford = false;
    bool pad_correction_active = false;
};

// Per-core byte sizes of the statically-allocated circular buffers used by the sharded
// group-norm program factory.
struct GroupNormShardedStaticCbSizes {
    uint32_t in_CB_size = 0;                // c_1  tilized input (and the c_30 untilize-out copy)
    uint32_t in2_CB_size = 0;               // c_2  scaler (and c_4 scaler-c when !welford)
    uint32_t in3_CB_size = 0;               // c_3  eps
    uint32_t in5_CB_size = 0;               // c_5  gamma
    uint32_t in6_CB_size = 0;               // c_6  beta
    uint32_t in_mask_CB_size = 0;           // c_7  input mask
    uint32_t in_negative_mask_CB_size = 0;  // c_14 negative mask
    uint32_t repack_CB_size = 0;            // c_11/c_12 repack
    uint32_t x_CB_size = 0;                 // c_13 x
    uint32_t ex_partial_CB_size = 0;        // c_8  ex_partial
    uint32_t ex_global_CB_size = 0;         // c_9/c_15 ex_global
    uint32_t ex2pe_CB_size = 0;             // c_17 ex2pe
    uint32_t single_tile_size = 0;          // c_10 ex_external
    uint32_t scalar_tile_size = 0;          // c_26 ones -- bf16 even on the legacy fp32 path
    uint32_t rowvalid_CB_size = 0;          // c_18, bf16, 1 tile, pad correction only
    uint32_t composed_mask_CB_size = 0;     // c_19, bf16, block_wt tiles, pad correction only

    // Total per-core L1 occupied by the static CB region.
    uint32_t total(const GroupNormShardedCbFlags& flags) const;
};

GroupNormShardedStaticCbSizes compute_sharded_gn_static_cb_sizes(
    const ttnn::Tensor& input,
    tt::tt_metal::DataType im_data_format,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    std::optional<tt::tt_metal::DataType> input_mask_dtype,
    std::optional<tt::tt_metal::DataType> negative_mask_dtype,
    bool use_welford,
    uint32_t num_groups);

}  // namespace ttnn::prim
