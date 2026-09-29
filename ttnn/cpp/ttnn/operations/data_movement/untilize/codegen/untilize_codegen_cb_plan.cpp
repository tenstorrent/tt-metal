// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_cb_plan.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

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

}  // namespace

ChosenCodegenCbPlan choose_codegen_cb_plan(
    const UntilizeCodegenOperationAttributes& attrs,
    const UntilizeCodegenTensorArgs& tensor_args,
    uint32_t reserved_l1_bytes_per_core) {
    const Tensor& input = tensor_args.input;
    auto out_spec = UntilizeCodegenDeviceOperation::compute_output_specs(attrs, tensor_args);
    DataType in_dtype = input.dtype();
    DataType out_dtype = out_spec.data_type();
    bool fp32 = needs_dst_accum(in_dtype);
    auto in_fmt = datatype_to_dataformat_converter(in_dtype);
    auto out_fmt = datatype_to_dataformat_converter(out_dtype);
    uint32_t tile_size_for_planning = std::max(tt::tile_size(in_fmt), tt::tile_size(out_fmt));
    uint64_t usable_l1 = ttnn::operations::data_movement::get_max_l1_space(input);
    usable_l1 = usable_l1 > reserved_l1_bytes_per_core ? usable_l1 - reserved_l1_bytes_per_core : 0;
    auto pages = pages_for_builder(input, fp32);
    auto depths = plan_cb_depths(usable_l1, pages.pages_per_unit, tile_size_for_planning, pages.block_units);
    if (!depths.has_value()) {
        return ChosenCodegenCbPlan{CodegenCbPlan::Native, std::nullopt};
    }
    return ChosenCodegenCbPlan{tier_from_depths(*depths, pages.pages_per_unit), depths};
}

}  // namespace ttnn::prim::untilize_codegen_detail
