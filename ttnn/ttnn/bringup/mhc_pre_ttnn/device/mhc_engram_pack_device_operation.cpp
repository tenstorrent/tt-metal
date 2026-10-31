// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_engram_pack_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

namespace {
constexpr uint32_t TILE_HW = 32;
constexpr uint32_t MAX_COLS_PER_BLOCK = 16;  // per unit: 2 x 16 data tiles in one read batch, 16 weight tiles
constexpr uint32_t UNIT_COST = 6;            // a unit's fixed work in column-tile equivalents (tuning estimate)
constexpr uint32_t MAX_PARTIALS = 64;        // partials per row (the root's CB holds one row's)
}  // namespace

MhcEngramPackPlan mhc_engram_pack_plan(const MhcEngramPackParams& prm, const MhcEngramPackInputs& in) {
    MhcEngramPackPlan p;
    p.n = prm.n;
    p.col_tiles = in.streams.logical_shape()[-1] / prm.n / TILE_HW;
    p.key_tiles = in.keys.padded_shape()[-1] / TILE_HW;
    const auto& padded = in.streams.padded_shape();
    uint32_t lead = 1;
    for (size_t i = 0; i + 2 < padded.rank(); ++i) {
        lead *= padded[i];
    }
    p.row_tiles = lead * (padded[-2] / TILE_HW);
    const auto grid = cut_grid(in.streams.device()->compute_with_storage_grid_size(), prm.max_grid);
    const uint32_t cores = grid.x * grid.y;
    // the block width that minimises the busiest core's work, (units) x (B columns + a per-unit overhead of about
    // UNIT_COST columns: the 3 re-lays, totals and the partial write), with at most MAX_PARTIALS partials per row
    // (ties: the wider block)
    uint32_t best_cost = 0, best_cb = 0;
    for (uint32_t cb = 1; cb <= std::min(MAX_COLS_PER_BLOCK, p.col_tiles); ++cb) {
        const uint32_t g = (p.col_tiles + cb - 1) / cb;
        if (p.n * g > MAX_PARTIALS && cb < std::min(MAX_COLS_PER_BLOCK, p.col_tiles)) {
            continue;
        }
        const uint32_t units = p.n * g * p.row_tiles;
        const uint32_t cost = ((units + cores - 1) / cores) * (cb + UNIT_COST);
        if (best_cb == 0 || cost <= best_cost) {
            best_cost = cost;
            best_cb = cb;
        }
    }
    p.cols_per_block = best_cb;
    p.groups = (p.col_tiles + best_cb - 1) / best_cb;
    p.units = p.n * p.groups * p.row_tiles;
    return p;
}

void MhcEngramPackDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& t) {
    for (const Tensor* x : {&t.streams, &t.keys, &t.weight}) {
        TT_FATAL(
            x->storage_type() == ttnn::StorageType::DEVICE, "mhc_pre_xing_pack: every operand must be on a device");
    }
}

void MhcEngramPackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

MhcEngramPackDeviceOperation::spec_return_value_t MhcEngramPackDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    auto layout = tt::tt_metal::TensorLayout(
        DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    const auto& s = t.streams.logical_shape();
    ttsl::SmallVector<uint32_t> dims;
    for (size_t i = 0; i + 1 < s.rank(); ++i) {
        dims.push_back(s[i]);
    }
    dims.push_back(TILE_HW);
    spec_return_value_t specs;
    specs.emplace_back(ttnn::Shape(dims), layout);
    const auto p = mhc_engram_pack_plan(attrs, t);
    specs.emplace_back(ttnn::Shape({1, 1, p.row_tiles * TILE_HW, p.n * p.groups * TILE_HW}), layout);  // partials
    return specs;
}

MhcEngramPackDeviceOperation::tensor_return_value_t MhcEngramPackDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    tensor_return_value_t out;
    for (const auto& spec : compute_output_specs(attrs, t)) {
        out.push_back(create_device_tensor(spec, t.streams.device()));
    }
    return out;
}

ttsl::hash::hash_t MhcEngramPackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (attrs.gate_eps.has_value()) {  // P.24: the gate epilogue's own programs (the off hashes unchanged)
        return tt::tt_metal::operation::hash_operation<MhcEngramPackDeviceOperation>(
            attrs.n,
            t.streams.tensor_spec(),
            t.keys.tensor_spec(),
            t.weight.tensor_spec(),
            attrs.max_grid.has_value() ? attrs.max_grid->x : 0u,
            attrs.max_grid.has_value() ? attrs.max_grid->y : 0u,
            std::bit_cast<uint32_t>(*attrs.gate_eps));
    }
    if (attrs.max_grid.has_value()) {  // FIX.15: the off hash is the one it always was
        return tt::tt_metal::operation::hash_operation<MhcEngramPackDeviceOperation>(
            attrs.n,
            t.streams.tensor_spec(),
            t.keys.tensor_spec(),
            t.weight.tensor_spec(),
            attrs.max_grid->x,
            attrs.max_grid->y);
    }
    return tt::tt_metal::operation::hash_operation<MhcEngramPackDeviceOperation>(
        attrs.n, t.streams.tensor_spec(), t.keys.tensor_spec(), t.weight.tensor_spec());
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {

std::vector<ttnn::Tensor> mhc_engram_pack(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& keys,
    const ttnn::Tensor& weight,
    uint32_t n,
    std::optional<tt::tt_metal::CoreCoord> max_grid,
    std::optional<float> gate_eps) {
    using OperationType = ttnn::operations::bringup::mhc_pre_ttnn::MhcEngramPackDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{.n = n, .max_grid = max_grid, .gate_eps = gate_eps};
    auto args = OperationType::tensor_args_t{.streams = streams, .keys = keys, .weight = weight};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim::bringup
