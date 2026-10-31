// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_stats_device_operation.hpp"
#include "mhc_post_given_output.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

namespace {
constexpr uint32_t TILE_HW = 32;
constexpr uint32_t MAX_COLS_PER_BLOCK = 8;  // W block in L1: n * 8 fp32 tiles, double-buffered (256 KB at n = 4)
constexpr uint32_t UNIT_COST = 4;           // a unit's fixed work in column-tile equivalents (tuning estimate)
constexpr uint32_t MAX_PARTIALS = 64;       // partials per row (the root's CB holds one row's)
}  // namespace

MhcPostStatsPlan mhc_post_stats_plan(
    const MhcPostStatsInputs& in, const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    MhcPostStatsPlan p;
    p.n = in.residual.logical_shape()[-1] / in.input.logical_shape()[-1];  // = post's last dim without offsets
    p.col_tiles = in.input.logical_shape()[-1] / TILE_HW;
    const auto& padded = in.input.padded_shape();
    uint32_t lead = 1;
    for (size_t i = 0; i + 2 < padded.rank(); ++i) {
        lead *= padded[i];
    }
    p.row_tiles = lead * (padded[-2] / TILE_HW);
    const auto grid = cut_grid(in.input.device()->compute_with_storage_grid_size(), max_grid);
    const uint32_t cores = grid.x * grid.y;
    // The block width that minimises the busiest core's work, (units) x (B columns + a per-unit overhead of about
    // UNIT_COST columns: the coefficient build, the partial and its write; measured 2x2, 1024 x 2560: B = 8 0.330 ms,
    // 4 0.352, 2 0.487, 1 0.909), with at most MAX_PARTIALS partials per row (ties: the wider block).
    uint32_t best_cost = 0, best_cb = 0;
    for (uint32_t cb = 1; cb <= std::min(MAX_COLS_PER_BLOCK, p.col_tiles); ++cb) {
        const uint32_t nb = (p.col_tiles + cb - 1) / cb;
        if (nb > MAX_PARTIALS && cb < std::min(MAX_COLS_PER_BLOCK, p.col_tiles)) {
            continue;
        }
        const uint32_t units = nb * p.row_tiles;
        const uint32_t per_core = (units + cores - 1) / cores;
        const uint32_t cost = per_core * (cb + UNIT_COST);
        if (best_cb == 0 || cost <= best_cost) {
            best_cost = cost;
            best_cb = cb;
        }
    }
    p.cols_per_block = best_cb;
    p.blocks = (p.col_tiles + best_cb - 1) / best_cb;
    p.units = p.blocks * p.row_tiles;
    return p;
}

void MhcPostStatsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& t) {
    for (const Tensor* x : {&t.input, &t.residual, &t.post, &t.comb, &t.weight}) {
        TT_FATAL(x->storage_type() == ttnn::StorageType::DEVICE, "mhc_post: every operand must be on a device");
    }
}

void MhcPostStatsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcPostStatsDeviceOperation::spec_return_value_t MhcPostStatsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    auto layout = [](DataType d) {
        return tt::tt_metal::TensorLayout(
            d, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    };
    const auto& x = t.residual;
    spec_return_value_t specs;
    specs.emplace_back(x.logical_shape(), layout(x.dtype()));
    const auto& s = x.logical_shape();
    ttsl::SmallVector<uint32_t> dims;
    for (size_t i = 0; i + 1 < s.rank(); ++i) {
        dims.push_back(s[i]);
    }
    dims.push_back(TILE_HW);
    specs.emplace_back(ttnn::Shape(dims), layout(DataType::FLOAT32));
    const auto p = mhc_post_stats_plan(t, attrs.max_grid);
    specs.emplace_back(
        ttnn::Shape({1, 1, p.row_tiles * TILE_HW, p.blocks * TILE_HW}), layout(DataType::FLOAT32));  // scratch
    return specs;
}

MhcPostStatsDeviceOperation::tensor_return_value_t MhcPostStatsDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto specs = compute_output_specs(attrs, t);
    const auto ins = {&t.input, &t.residual, &t.post, &t.comb, &t.weight};
    tensor_return_value_t out;
    out.push_back(given_or_new(t.output, specs[0], t.residual, ins, "output_tensor"));
    out.push_back(given_or_new(t.stats_output, specs[1], t.residual, ins, "stats_output"));
    out.push_back(given_or_new(t.stats_scratch, specs[2], t.residual, ins, "stats_scratch"));
    return out;
}

ttsl::hash::hash_t MhcPostStatsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (attrs.max_grid.has_value()) {  // FIX.15: the hash below, then the grid; the off hashes are unchanged
        auto off = attrs;
        off.max_grid.reset();
        return tt::tt_metal::operation::hash_operation<MhcPostStatsDeviceOperation>(
            compute_program_hash(off, t), attrs.max_grid->x, attrs.max_grid->y);
    }
    const auto& cc = attrs.compute_config;
    if (attrs.post_col != 0 || attrs.comb_col != 0) {  // P.19: column offsets, a separate program; off: below
        return tt::tt_metal::operation::hash_operation<MhcPostStatsDeviceOperation>(
            static_cast<uint32_t>(cc.math_fidelity),
            cc.fp32_dest_acc_en,
            cc.math_approx_mode,
            attrs.comb_transposed,
            t.input.tensor_spec(),
            t.residual.tensor_spec(),
            t.post.tensor_spec(),
            t.comb.tensor_spec(),
            t.weight.tensor_spec(),
            attrs.post_col,
            attrs.comb_col);
    }
    return tt::tt_metal::operation::hash_operation<MhcPostStatsDeviceOperation>(
        static_cast<uint32_t>(cc.math_fidelity),
        cc.fp32_dest_acc_en,
        cc.math_approx_mode,
        attrs.comb_transposed,
        t.input.tensor_spec(),
        t.residual.tensor_spec(),
        t.post.tensor_spec(),
        t.comb.tensor_spec(),
        t.weight.tensor_spec());
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn

namespace ttnn::prim::bringup {

std::vector<ttnn::Tensor> mhc_post_stats(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& weight,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    bool comb_transposed,
    uint32_t post_col,
    uint32_t comb_col,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& stats_output,
    const std::optional<ttnn::Tensor>& stats_scratch,
    std::optional<tt::tt_metal::CoreCoord> max_grid) {
    using OperationType = ttnn::operations::bringup::mhc_post_ttnn::MhcPostStatsDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .compute_config = compute_config,
        .comb_transposed = comb_transposed,
        .post_col = post_col,
        .comb_col = comb_col,
        .max_grid = max_grid};
    auto args = OperationType::tensor_args_t{
        .input = input,
        .residual = residual,
        .post = post,
        .comb = comb,
        .weight = weight,
        .output = output_tensor,
        .stats_output = stats_output,
        .stats_scratch = stats_scratch};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim::bringup
