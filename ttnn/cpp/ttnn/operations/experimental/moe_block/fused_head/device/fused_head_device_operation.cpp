// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_head_device_operation.hpp"

#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace fused_head_op_detail {

void check_dram(const Tensor& t, const char* what) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr, "fused_head: {} must be on device", what);
    TT_FATAL(t.memory_config().buffer_type() == BufferType::DRAM, "fused_head: {} must be in DRAM", what);
}

std::vector<ttnn::MeshCoordinate> coords_of(const ttnn::MeshCoordinateRangeSet& set) {
    std::vector<ttnn::MeshCoordinate> out;
    for (const auto& range : set.ranges()) {
        for (const auto& c : range) {
            out.push_back(c);
        }
    }
    return out;
}

}  // namespace fused_head_op_detail
using namespace fused_head_op_detail;

ttsl::hash::hash_t FusedHeadDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& c = attrs.config;
    return tt::tt_metal::operation::hash_operation<FusedHeadDeviceOperation>(
        attrs.plan_key,
        c.lay.GY,
        c.lay.MT,
        c.lay.NCH,
        c.lay.KB,
        c.lay.GX,
        c.lay.compact,
        c.lay.xdram,
        c.lay.NIT,
        c.shape.rows,
        c.shape.A,
        c.fidelity,
        c.w_dtype,
        c.l1_free,
        c.cb_floor,
        args.x.tensor_spec(),
        args.w.tensor_spec(),
        args.w_router.tensor_spec(),
        args.actual_start.has_value(),
        attrs.loop);
}

FusedHeadDeviceOperation::program_factory_t FusedHeadDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return FusedHeadFactory{};
}

void FusedHeadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(attrs.model != nullptr, "fused_head: the host model is missing");
    const auto s = head_tensor_shapes(*attrs.model);
    check_dram(args.x, "x");
    const auto& xs = args.x.logical_shape();
    TT_FATAL(
        xs[-2] == attrs.config.shape.rows && xs[-1] == attrs.config.shape.H / attrs.config.shape.G,
        "fused_head: x must be [.., {}, {}] a chip, got {}",
        attrs.config.shape.rows,
        attrs.config.shape.H / attrs.config.shape.G,
        xs);
    TT_FATAL(args.x.layout() == Layout::TILE && args.x.dtype() == DataType::BFLOAT16, "fused_head: x is bf16 TILE");
    TT_FATAL(
        attrs.config.lay.xdram == args.x_full.has_value(),
        "fused_head: the DRAM fallback layout takes x_full (the gathered x), the resident ones do not");
    for (const auto* t :
         {&args.w,
          &args.w_router,
          &args.router_bias,
          &args.router_index,
          &args.y,
          &args.rows,
          &args.xbuf,
          &args.route,
          &args.slots,
          &args.fslots,
          &args.counts,
          &args.regions,
          &args.token_index,
          &args.table,
          &args.plan}) {
        check_dram(*t, "an argument");
    }
    TT_FATAL(
        args.actual_start.has_value() == args.actual_end.has_value(),
        "fused_head: a padded chunk takes both actual_start and actual_end");
    for (const auto* v : {&args.actual_start, &args.actual_end}) {
        if (*v) {
            check_dram(**v, "actual_start / actual_end");
            TT_FATAL(
                (*v)->dtype() == DataType::UINT32 && (*v)->logical_volume() == 1 && !(*v)->is_sharded(),
                "fused_head: actual_start / actual_end are one uint32 each (DRAM interleaved), got {} of {} elements",
                (*v)->dtype(),
                (*v)->logical_volume());
        }
    }
    TT_FATAL(
        args.y.logical_shape()[-2] == s.y_rows && args.y.logical_shape()[-1] == s.y_cols,
        "fused_head: y must be [{}, {}]",
        s.y_rows,
        s.y_cols);
}

void FusedHeadDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

FusedHeadDeviceOperation::spec_return_value_t FusedHeadDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y.tensor_spec();
}

FusedHeadDeviceOperation::tensor_return_value_t FusedHeadDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y;
}

HeadBuffers head_buffers(const FusedHeadParams& attrs, const FusedHeadInputs& a) {
    HeadBuffers b;
    TT_FATAL(SRC_WORD0 + attrs.words.size() <= kHeadMaxSrc, "fused_head: {} persistent words", attrs.words.size());
    auto set = [&](HeadSrc s, const Tensor* t) {
        if (t != nullptr) {
            b.buffers[s] = t->buffer();
            b.addresses[s] = static_cast<uint32_t>(t->buffer()->address());
        }
    };
    set(SRC_X, &a.x);
    set(SRC_X_FULL, a.x_full ? &*a.x_full : nullptr);
    set(SRC_W, &a.w);
    set(SRC_Y, &a.y);
    set(SRC_ROWS, &a.rows);
    set(SRC_XBUF, &a.xbuf);
    set(SRC_ROUTE, &a.route);
    set(SRC_RW, &a.w_router);
    set(SRC_RB, &a.router_bias);
    set(SRC_RIX, &a.router_index);
    set(SRC_SLOTS, &a.slots);
    set(SRC_FSLOTS, &a.fslots);
    set(SRC_COUNTS, &a.counts);
    set(SRC_REGIONS, &a.regions);
    set(SRC_TOKEN_INDEX, &a.token_index);
    set(SRC_TABLE, &a.table);
    set(SRC_PLAN, &a.plan);
    set(SRC_ARENA, &attrs.arena);
    set(SRC_START, a.actual_start ? &*a.actual_start : nullptr);
    set(SRC_END, a.actual_end ? &*a.actual_end : nullptr);
    for (size_t i = 0; i < attrs.words.size(); ++i) {
        b.addresses[SRC_WORD0 + i] = attrs.words[i];
    }
    b.loops = attrs.loops;
    b.loop = attrs.loop % attrs.loops;
    b.split = attrs.loop / attrs.loops;
    return b;
}

tt::tt_metal::WorkloadDescriptor FusedHeadFactory::create_workload_descriptor(
    const FusedHeadParams& attrs,
    const FusedHeadInputs& args,
    Tensor& /*output*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    std::vector<HeadPatches> patches;
    auto w = build_head_workload(*attrs.model, head_buffers(attrs, args), coords_of(tensor_coords), &patches);
    std::lock_guard lock(attrs.patches->mutex);
    if (attrs.patches->chips.empty()) {
        attrs.patches->chips = std::move(patches);
    }
    return w;
}

void FusedHeadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const FusedHeadParams& attrs,
    const FusedHeadInputs& args,
    Tensor& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    TT_FATAL(coord.has_value(), "fused_head: one program a chip");
    const HeadBuffers b = head_buffers(attrs, args);
    auto& store = *attrs.patches;
    HeadLastAddresses* last = nullptr;
    {
        // (programs another resources object built: where their address args sit, once - every chip's)
        std::lock_guard lock(store.mutex);
        last = &store.last[&program];  // (a map node: stable while we patch)
        if (store.chips.empty()) {
            const auto& m = *attrs.model;
            std::vector<ttnn::MeshCoordinate> all;
            for (uint32_t i = 0; i < attrs.config.shape.N(); ++i) {
                all.emplace_back(i / attrs.config.shape.G, i % attrs.config.shape.G);
            }
            build_head_workload(m, b, all, &store.chips);
        }
    }
    patch_head_program(program, store.chips[(*coord)[0] * attrs.config.shape.G + (*coord)[1]], b, *last);
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head

namespace ttnn::prim {

Tensor moe_block_fused_head(
    const ttnn::operations::experimental::moe_block::fused_head::FusedHeadParams& params,
    ttnn::operations::experimental::moe_block::fused_head::FusedHeadInputs inputs) {
    using OperationType = ttnn::operations::experimental::moe_block::fused_head::FusedHeadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, std::move(inputs));
}

}  // namespace ttnn::prim
