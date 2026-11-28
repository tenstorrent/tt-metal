// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_device_operation.hpp"

#include <algorithm>
#include <string>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "grouped_matmul_planner.hpp"

namespace ttnn::operations::grouped_matmul {

using namespace tt::tt_metal;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw GroupedMatmulAssertion(message);
    }
}

// program() / bank_program(): rt clamped to MT, then rt x pn_max within DEST (full-sync doubles it).
void check_dest_fit(const GroupedMatmulParams& attrs, uint32_t mt, uint32_t pn_max) {
    const uint32_t half = attrs.fp32_dest_acc_en ? 4 : 8;
    const uint32_t rt = std::max<uint32_t>(1, std::min(attrs.rt, mt));
    const uint32_t dst = half * (rt * pn_max > half ? 2 : 1);
    require(
        rt * pn_max <= dst,
        fmt::format("grouped_matmul: rt {} x {} column tiles exceed DEST ({} tiles)", rt, pn_max, dst));
}

// The bank path (FIX.42): a core whose rt x column tiles exceed DEST runs its columns in sub-blocks of
// bank_n_sub() tiles (GMM_N_SUB); only rt itself over DEST is refused there.
void check_bank_dest_fit(const GroupedMatmulParams& attrs, uint32_t mt, uint32_t pn_max) {
    const uint32_t rt = std::max<uint32_t>(1, std::min(attrs.rt, mt));
    require(
        bank_n_sub(attrs.fp32_dest_acc_en, rt, pn_max) != kBankNSubNone,
        fmt::format("grouped_matmul: rt {} x {} column tiles exceed DEST (bank path: rt alone over DEST)", rt, pn_max));
}

void check_depth(uint32_t depth) { require(depth >= 2, fmt::format("grouped_matmul: depth {} (>= 2)", depth)); }

}  // namespace

GroupedMatmulDeviceOperation::program_factory_t GroupedMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return GroupedMatmulProgramFactory{};
}

void GroupedMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& xs = t.x.logical_shape();
    const auto& ws = t.w.logical_shape();
    const uint32_t a = tile_size(t.x.dtype());
    const uint32_t b = tile_size(t.w.dtype());
    const uint32_t o = tile_size(attrs.output_dtype);
    const uint32_t f32 = tile_size(DataType::FLOAT32);

    if (!attrs.bank_layout.has_value()) {
        InterleavedPlanInputs in;
        in.x_rank = xs.rank();
        in.w_rank = ws.rank();
        if (in.x_rank == 4 && in.w_rank == 4) {
            in.x0 = xs[0], in.x1 = xs[1], in.x2 = xs[2], in.x3 = xs[3];
            in.w0 = ws[0], in.w1 = ws[1], in.w2 = ws[2], in.w3 = ws[3];
        }
        in.a_tile_bytes = a;
        in.b_tile_bytes = b;
        in.o_tile_bytes = o;
        in.f32_tile_bytes = f32;
        in.grid_x = attrs.grid_x;
        in.grid_y = attrs.grid_y;
        in.block_w = attrs.block_w;
        const InterleavedPlan* plan = solve_interleaved_plan(in);
        require(plan != nullptr, fmt::format("grouped_matmul: no layout for {} x {}", xs, ws));
        check_depth(attrs.depth);
        check_dest_fit(attrs, plan->MT, plan->pn_max());
        return;
    }

    const BankLayout& layout = *attrs.bank_layout;
    require(xs.rank() == 4, fmt::format("grouped_matmul: x {}", xs));
    const uint32_t g = xs[1];
    const uint32_t m = xs[2];
    const uint32_t k = xs[3];
    require(
        g == layout.G && k / tt::constants::TILE_WIDTH == layout.KT && k % tt::constants::TILE_WIDTH == 0,
        fmt::format("grouped_matmul: x {} vs layout G {} KT {}", xs, layout.G, layout.KT));
    uint32_t pn_max = 0;
    for (const auto& s : layout.slots) {
        pn_max = std::max(pn_max, s.pn);
    }
    const uint32_t mt = m / tt::constants::TILE_HEIGHT;
    const uint32_t bw = solve_bank_block_w(mt, layout.KT, pn_max, a, b, o, f32, attrs.block_w);
    require(bw != 0 && layout.KT % bw == 0, fmt::format("grouped_matmul: bank path, block_w {}", attrs.block_w));
    check_depth(attrs.depth);
    require(
        attrs.w_inflight >= 1 && attrs.w_inflight <= std::min<uint32_t>(attrs.depth, 8),
        fmt::format("grouped_matmul: w_inflight {} (1 .. min(depth, 8))", attrs.w_inflight));
    require(
        attrs.x_ahead <= std::min<uint32_t>(3, attrs.depth - 1),
        fmt::format("grouped_matmul: x_ahead {} (0 .. min(3, depth - 1))", attrs.x_ahead));
    check_bank_dest_fit(attrs, mt, pn_max);
}

void GroupedMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t GroupedMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // Every attribute (all structural) and the specs of x and w. The caller's output_tensor, present or absent, is
    // left out, as the Python op's program key left it out: the entry refuses one whose dtype, layout or shape differs
    // from the output spec, and its memory config is attrs.output_memory_config, so with or without it the program is
    // the same; override_runtime_arguments re-applies its address (the writer's Y_ADDR) on every hit.
    return tt::tt_metal::operation::hash_operation<GroupedMatmulDeviceOperation>(
        attrs, t.x.tensor_spec(), t.w.tensor_spec());
}

GroupedMatmulDeviceOperation::spec_return_value_t GroupedMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& xs = t.x.logical_shape();
    const uint32_t n =
        attrs.bank_layout.has_value() ? attrs.bank_layout->NT * tt::constants::TILE_WIDTH : t.w.logical_shape()[3];
    return TensorSpec(
        ttnn::Shape({1, xs[1], xs[2], n}),
        TensorLayout(attrs.output_dtype, PageConfig(Layout::TILE), attrs.output_memory_config));
}

GroupedMatmulDeviceOperation::tensor_return_value_t GroupedMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.x.device());
}

}  // namespace ttnn::operations::grouped_matmul

namespace ttnn::prim {

Tensor grouped_matmul(
    const Tensor& x,
    const Tensor& w,
    const std::optional<Tensor>& output,
    const ttnn::operations::grouped_matmul::GroupedMatmulParams& params) {
    using OperationType = ttnn::operations::grouped_matmul::GroupedMatmulDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.x = x, .w = w, .output = output});
}

}  // namespace ttnn::prim
