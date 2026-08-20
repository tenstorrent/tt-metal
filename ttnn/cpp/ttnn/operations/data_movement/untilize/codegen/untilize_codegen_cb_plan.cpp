// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_cb_plan.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::untilize_codegen_detail {

uint32_t compute_block_ct_dim(uint32_t wt, bool fp32) {
    uint32_t max_bct = fp32 ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (wt % bct == 0) {
            return bct;
        }
    }
    return 1;
}

uint32_t choose_2d_ncol(uint32_t total_tile_rows, uint32_t wt, uint32_t valid_cores) {
    if (total_tile_rows >= valid_cores || wt < 2) {
        return 1;
    }
    uint32_t max_ncol = std::min(valid_cores / total_tile_rows, wt);
    uint32_t best = 1;
    for (uint32_t d = 2; d <= max_ncol; ++d) {
        if (wt % d == 0) {
            best = d;
        }
    }
    return best;
}

std::optional<CbPlan> plan_cb_depths(
    uint64_t usable_l1, uint32_t pages_per_unit, uint32_t page_size, uint32_t block_units) {
    const uint64_t pages = pages_per_unit;
    const uint64_t tile_bytes = page_size;
    const uint64_t double_both = (2 * pages + 2 * pages) * tile_bytes;
    const uint64_t double_in = (2 * pages + pages) * tile_bytes;
    const uint64_t single_both = (pages + pages) * tile_bytes;
    if (double_both <= usable_l1) {
        return CbPlan{2 * pages_per_unit, 2 * pages_per_unit, pages_per_unit};
    }
    if (double_in <= usable_l1) {
        return CbPlan{2 * pages_per_unit, pages_per_unit, pages_per_unit};
    }
    if (single_both <= usable_l1) {
        return CbPlan{pages_per_unit, pages_per_unit, block_units};
    }
    return std::nullopt;
}

namespace {

bool needs_dst_accum(DataType dtype) {
    return dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

struct PaddedGrid {
    uint32_t wt;
    uint32_t total_tile_rows;
    bool tile_aligned;
};

PaddedGrid padded_grid(const Tensor& input) {
    const auto& padded_shape = input.padded_shape();
    const auto rank = padded_shape.rank();
    const uint32_t w = padded_shape[-1];
    const uint32_t h = padded_shape[-2];
    const uint32_t batch_dims = rank > 2 ? rank - 2 : 0;
    const uint32_t nc = std::accumulate(
        padded_shape.begin(), padded_shape.begin() + batch_dims, uint32_t{1}, std::multiplies<uint32_t>{});
    uint32_t wt = w / TILE_WIDTH;
    uint32_t ht = h / TILE_HEIGHT;
    const auto& logical_shape = input.logical_shape();
    bool tile_aligned = logical_shape[-2] % TILE_HEIGHT == 0 && logical_shape[-1] % TILE_WIDTH == 0;
    return PaddedGrid{wt, nc * ht, tile_aligned};
}

struct PagesAndBlock {
    uint32_t pages_per_unit;
    uint32_t block_units;
};

PagesAndBlock pages_for_builder(const Tensor& input, bool fp32) {
    auto g = padded_grid(input);
    auto* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t valid_cores = static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);

    if (!g.tile_aligned) {
        return {g.wt, compute_block_ct_dim(g.wt, fp32)};
    }
    if (g.total_tile_rows == 1 && g.wt > 1) {
        auto [_num_cores, _core_range, cg1, cg2, tpc1, tpc2] =
            tt::tt_metal::split_work_to_cores(grid, g.wt, /*row_wise=*/true);
        uint32_t max_tpc = std::max(tpc1, cg2.empty() ? 0u : tpc2);
        return {max_tpc, compute_block_ct_dim(max_tpc, fp32)};
    }
    if (g.wt > 1) {
        uint32_t ncol = choose_2d_ncol(g.total_tile_rows, g.wt, valid_cores);
        if (ncol >= 2) {
            uint32_t tpc = g.wt / ncol;
            return {tpc, compute_block_ct_dim(tpc, fp32)};
        }
    }
    return {g.wt, compute_block_ct_dim(g.wt, fp32)};
}

CodegenCbPlan tier_from_depths(const CbPlan& plan, uint32_t pages_per_unit) {
    if (plan.cb_in_depth == 2 * pages_per_unit && plan.cb_out_depth == 2 * pages_per_unit) {
        return CodegenCbPlan::DoubleBoth;
    }
    if (plan.cb_in_depth == 2 * pages_per_unit && plan.cb_out_depth == pages_per_unit) {
        return CodegenCbPlan::DoubleIn;
    }
    return CodegenCbPlan::SingleBoth;
}

NativeCacheIdentity identity_from_ncores(
    const ttnn::NcoresWHsb& sb, uint32_t width_tiles, uint32_t height_tiles) {
    NativeCacheIdentity id;
    id.enough_space_height = false;
    id.split_valid = true;
    id.ncores = sb.ncores;
    id.nblocks_per_core = sb.nblocks_per_core;
    id.single_block_size = sb.single_block_size;
    if (id.single_block_size == 0) {
        id.split_valid = false;
        return id;
    }
    uint32_t total_blocks_width = sb.total_blocks_width;
    uint32_t total_blocks_height = sb.total_blocks_height;
    id.full_cores_per_row = width_tiles / id.single_block_size;
    id.has_cliff_row = (id.full_cores_per_row < total_blocks_width);
    id.full_cores_per_col = height_tiles / id.single_block_size;
    id.has_cliff_col = (id.full_cores_per_col < total_blocks_height);
    id.single_block_size_cliff_row = width_tiles - (id.full_cores_per_row * id.single_block_size);
    id.single_block_size_cliff_col = height_tiles - (id.full_cores_per_col * id.single_block_size);
    id.single_sub_block_size = sb.single_sub_block_size;
    return id;
}

NativeCacheIdentity block_split_identity(
    uint32_t grid_area,
    uint32_t nblocks,
    uint32_t width_tiles,
    uint32_t height_tiles,
    uint32_t cb_block_size_limit) {
    auto wh = ttnn::compute_ncores_wh(grid_area, nblocks, width_tiles, height_tiles);
    ttnn::NcoresWHsb sb{
        wh.ncores,
        wh.nblocks_per_core,
        wh.total_blocks_width,
        wh.total_blocks_height,
        wh.single_block_size,
        wh.single_block_size};
    if (cb_block_size_limit >= 1 && wh.single_block_size > cb_block_size_limit) {
        auto maybe = ttnn::try_compute_ncores_wh_sb(
            grid_area, nblocks, width_tiles, height_tiles, cb_block_size_limit);
        if (!maybe.has_value()) {
            NativeCacheIdentity sentinel;
            sentinel.enough_space_height = false;
            sentinel.split_valid = false;
            return sentinel;
        }
        sb = *maybe;
    }
    return identity_from_ncores(sb, width_tiles, height_tiles);
}

}  // namespace

ChosenCodegenCbPlan choose_codegen_cb_plan(
    const UntilizeCodegenOperationAttributes& attrs, const UntilizeCodegenTensorArgs& tensor_args) {
    const Tensor& input = tensor_args.input;
    auto out_spec = UntilizeCodegenDeviceOperation::compute_output_specs(attrs, tensor_args);
    DataType in_dtype = input.dtype();
    DataType out_dtype = out_spec.data_type();
    bool fp32 = needs_dst_accum(in_dtype);
    auto in_fmt = datatype_to_dataformat_converter(in_dtype);
    auto out_fmt = datatype_to_dataformat_converter(out_dtype);
    uint32_t tile_size_for_planning = std::max(tt::tile_size(in_fmt), tt::tile_size(out_fmt));
    uint64_t usable_l1 = ttnn::operations::data_movement::get_max_l1_space(input);
    auto pages = pages_for_builder(input, fp32);
    auto depths = plan_cb_depths(usable_l1, pages.pages_per_unit, tile_size_for_planning, pages.block_units);
    if (!depths.has_value()) {
        return ChosenCodegenCbPlan{CodegenCbPlan::Native, std::nullopt};
    }
    return ChosenCodegenCbPlan{tier_from_depths(*depths, pages.pages_per_unit), depths};
}

NativeCacheIdentity native_cache_identity(
    const UntilizeCodegenOperationAttributes& attrs,
    const UntilizeCodegenTensorArgs& tensor_args,
    CodegenCbPlan plan) {
    (void)attrs;
    NativeCacheIdentity zeros;
    if (plan != CodegenCbPlan::Native) {
        return zeros;
    }

    namespace dm = ttnn::operations::data_movement;
    const Tensor& input = tensor_args.input;
    auto in_fmt = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(in_fmt);
    uint32_t num_tiles_per_row = input.padded_shape()[-1] / TILE_WIDTH;
    const bool enough_space_height =
        dm::is_enough_space(input, single_tile_size, single_tile_size, num_tiles_per_row);
    if (enough_space_height) {
        NativeCacheIdentity id;
        id.enough_space_height = true;
        return id;
    }

    auto out_spec = UntilizeCodegenDeviceOperation::compute_output_specs(attrs, tensor_args);
    auto out_fmt = datatype_to_dataformat_converter(out_spec.data_type());
    uint32_t output_single_tile_size = tt::tile_size(out_fmt);
    uint32_t max_l1_size = dm::get_max_l1_space(input);
    uint32_t denom = single_tile_size + output_single_tile_size;
    uint32_t cb_block_size_limit = denom == 0 ? 0 : max_l1_size / denom;

    const auto& logical_shape = input.logical_shape();
    const bool tile_aligned = logical_shape[-2] % TILE_HEIGHT == 0 && logical_shape[-1] % TILE_WIDTH == 0;
    auto* device = input.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    if (tile_aligned) {
        uint32_t a_tile_width = input.tensor_spec().tile().get_width();
        uint32_t a_tile_height = input.tensor_spec().tile().get_height();
        uint32_t width_tiles = input.padded_shape()[-1] / a_tile_width;
        uint32_t height_tiles = input.padded_shape()[-2] / a_tile_height;
        uint32_t nblocks = (input.padded_shape()[-1] * input.padded_shape()[-2]) / (a_tile_height * a_tile_width);
        uint32_t grid_area = static_cast<uint32_t>(grid_size.x) * static_cast<uint32_t>(grid_size.y);
        return block_split_identity(grid_area, nblocks, width_tiles, height_tiles, cb_block_size_limit);
    }

    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet available_grid(default_cores);
    uint32_t width_tiles = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t height_tiles = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t nblocks =
        (input.padded_shape()[-1] * input.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);
    return block_split_identity(
        available_grid.num_cores(), nblocks, width_tiles, height_tiles, cb_block_size_limit);
}

}  // namespace ttnn::prim::untilize_codegen_detail
