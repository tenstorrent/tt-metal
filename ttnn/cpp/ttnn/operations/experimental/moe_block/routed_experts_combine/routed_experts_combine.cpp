// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "device/routed_experts_combine_device_operation.hpp"
#include "ttnn/operations/experimental/moe_block/common/l1_budget.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_program.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert.hpp"
#include "ttnn/operations/experimental/moe_block/sparse_dispatch_2d/device/sparse_dispatch_2d_plan.hpp"
#include "ttnn/operations/experimental/moe_block/sparse_dispatch_2d/sparse_dispatch_2d.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_detail {

using tt::tt_metal::CoreCoord;
namespace fre = flat_routed_expert;

// What one resources object keeps for the combine across calls: the plans, the persistent tensors, the arena's spec.
struct CombineState {
    fre::FlatRoutedExpertConfig flat_config;
    fre::FlatLayoutOptions layout;  // (as the caller passed it)
    fre::FlatRoutedExpertResources flat;
    CombineConfig combine;
    std::vector<CoreCoord> cells;
    FlatHandoff handoff;
    std::shared_ptr<const CombinePlan> plan;
    ttnn::Tensor out, bar, hcache;
    std::optional<ttnn::Tensor> lut;
    std::optional<tt::tt_metal::TensorSpec> arena_spec;
    std::optional<ttnn::Tensor> y;  // the flat expert's y (never written with the L1 hand-off)
    uint32_t rows_cap = 0;
};

tt::tt_metal::TensorSpec rm_spec(const ttnn::Shape& shape, DataType dt, const MemoryConfig& mc) {
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), mc));
}

fre::FlatRoutedExpertConfig base_flat_config(
    MoEBlockResources& res, uint32_t m, uint32_t activation, const fre::FlatLayoutOptions& layout) {
    const auto& dims = res.dims();
    const uint32_t N = res.topology().num_chips();
    TT_FATAL(dims.num_experts % N == 0, "routed_experts_combine: {} experts over {} chips", dims.num_experts, N);
    auto cfg = fre::make_flat_routed_expert_config(
        dims.routed_width(),
        dims.routed_expert_hidden_dim,
        dims.num_experts / N,
        dims.num_experts,
        m,
        res.dtypes().routed_weights == DataType::BFLOAT8_B,
        1,
        layout);
    cfg.activation = activation;
    cfg.down_fp32 = true;
    return cfg;
}

// L1 a core the static circular buffers of the flat program and the dispatch (its manifest rings and CBs on the port
// cores, at this row group) take: the larger.
uint32_t static_cb_bytes(MoEBlockResources& res, const fre::FlatRoutedExpertPlan& fp, uint32_t group) {
    const auto& topo = res.topology();
    const uint32_t N = topo.num_chips(), S = res.config().chunk_tokens / N, W = res.dims().routed_width();
    const auto ml = manifest_layout(topo.R, topo.C, topo.wrap_x, topo.wrap_y, S);
    return std::max(
        align64(fp.static_l1_bytes), align64(ml.total_entries * 16ull) + dispatch_static_cb_bytes(N, S, W, group));
}

// The cells the flat plan leaves (k3 combine_cells), row-major.
std::vector<CoreCoord> free_cells(MoEBlockResources& res, const fre::FlatRoutedExpertPlan& fp) {
    const auto grid = res.mesh_device().compute_with_storage_grid_size();
    const auto taken = fp.arena_cores();
    std::vector<CoreCoord> out;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (std::none_of(taken.begin(), taken.end(), [&](const CoreCoord& c) { return c.x == x && c.y == y; })) {
                out.emplace_back(x, y);
            }
        }
    }
    return out;
}

// The block's L1 plan's group and head ports when the resources hold one, else the dispatch's own default and none.
std::pair<uint32_t, uint32_t> block_group_and_head_port(const MoEBlockResources& res) {
    const BlockL1Plan* l1 = block_l1_plan(res);
    return {sparse_dispatch_2d::dispatch_group(res), l1 ? l1->head_port_cb : 0u};
}

}  // namespace routed_experts_combine_detail

CombineL1Plan plan_routed_experts_combine_l1(
    MoEBlockResources& res,
    uint32_t m,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t group,
    uint32_t head_port_cb) {
    using namespace routed_experts_combine_detail;
    auto& mesh = res.mesh_device();
    const auto& cfg = res.config();
    const uint32_t N = res.topology().num_chips();
    TT_FATAL(cfg.chunk_tokens % N == 0, "routed_experts_combine: chunk {} over {} chips", cfg.chunk_tokens, N);
    // The flat expert as the caller gave it when its plan fits the arena the combine gets, else with that arena as
    // its budget (FlatLayoutOptions.arena_bytes: the flat plan fits it) - a smaller plan leaves the arena at least as
    // large, so the fitted plan still fits.
    CombineL1Plan out;
    out.flat = base_flat_config(res, m, activation, layout);
    out.arena_bytes = combine.arena_bytes;
    std::shared_ptr<const flat_routed_expert::FlatRoutedExpertPlan> fp;
    for (uint32_t i = 0; i < 4; ++i) {
        fp = fre::flat_routed_expert_plan(mesh, out.flat);
        out.static_bytes = static_cb_bytes(res, *fp, group);
        if (combine.arena_bytes == 0) {  // (the largest free L1 block less what stays below the arena)
            const uint32_t reserve = exchange_arena_reserve(std::max(out.static_bytes, head_port_cb));
            const auto stats = mesh.allocator()->get_statistics(tt::tt_metal::BufferType::L1);
            out.arena_bytes = exchange_arena_bytes(static_cast<uint32_t>(stats.largest_free_block_bytes), reserve);
        }
        if (fp->arena_tiles * 2048 <= out.arena_bytes || layout.arena_bytes > 0) {
            break;  // (a caller's own budget is kept: checked below)
        }
        out.flat.layout.arena_bytes = static_cast<int32_t>(out.arena_bytes);
    }
    TT_FATAL(
        out.arena_bytes >= fp->arena_tiles * 2048,
        "routed_experts_combine: the arena {} KB < the flat expert's {} KB",
        out.arena_bytes >> 10,
        (fp->arena_tiles * 2048) >> 10);
    out.cells = free_cells(res, *fp);
    CombinePlanInputs in;
    in.S = cfg.chunk_tokens / N;
    in.W = res.dims().routed_width();
    in.cells = out.cells;
    in.n_acc = combine.n_acc;
    in.mode = combine.mode;
    in.depth = combine.depth;
    in.arena_bytes = out.arena_bytes;
    out.plan = make_combine_plan(mesh, res, in);
    return out;
}

namespace routed_experts_combine_detail {

CombineState make_state(
    MoEBlockResources& res,
    uint32_t m,
    uint32_t activation,
    const fre::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t rows_cap) {
    auto& mesh = res.mesh_device();
    const uint32_t N = res.topology().num_chips();
    CombineState s;
    s.combine = combine;
    s.layout = layout;
    // (in the block: its L1 plan's dispatch group and the head's ports below the shared arena)
    const auto [group, head_port] = block_group_and_head_port(res);
    const CombineL1Plan l1 = plan_routed_experts_combine_l1(res, m, activation, layout, combine, group, head_port);
    s.flat_config = l1.flat;
    s.flat = fre::ensure_flat_routed_expert_resources(res, s.flat_config);
    s.flat_config = fre::with_combine_overlap(s.flat_config, *s.flat.plan, true);
    s.cells = l1.cells;
    s.handoff.writers = fre::flat_y_writers(*s.flat.plan);
    s.plan = l1.plan;
    const uint32_t S = s.plan->S, W = s.plan->W;

    // the call's arena: every worker core, the combine's bytes then the writers' words (rows of 64 B)
    const auto grid = mesh.compute_with_storage_grid_size();
    const uint32_t rows = l1.arena_bytes / 64 + kWriterWordsRows;
    const CoreRangeSet all(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    s.arena_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({static_cast<uint32_t>(grid.x * grid.y) * rows, 32}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig(
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                tt::tt_metal::ShardSpec(all, {rows, 32}, tt::tt_metal::ShardOrientation::ROW_MAJOR))));
    s.bar = res.persistent_words("routed_experts_combine/bar", all, 16);
    s.hcache = res.persistent_tensor(
        "routed_experts_combine/hcache",
        rm_spec(ttnn::Shape({8, hcache_page_bytes(N) / 4}), DataType::UINT32, DRAM_MEMORY_CONFIG));
    s.out = res.persistent_tensor(
        "routed_experts_combine/out", rm_spec(ttnn::Shape({1, 1, S, W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
    s.rows_cap = rows_cap;
    s.y = res.persistent_tensor(
        "routed_experts_combine/y", rm_spec(ttnn::Shape({rows_cap, W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
    if (s.plan->perchip) {
        const auto& lut16 = s.plan->lut;
        std::vector<uint32_t> words((lut16.size() + 1) / 2, 0);
        for (size_t i = 0; i < lut16.size(); ++i) {
            words[i / 2] |= static_cast<uint32_t>(lut16[i]) << (16 * (i % 2));
        }
        const auto spec =
            rm_spec(ttnn::Shape({1, static_cast<uint32_t>(words.size())}), DataType::UINT32, DRAM_MEMORY_CONFIG);
        s.lut = res.plan<ttnn::Tensor>(
            "routed_experts_combine/lut", [&] { return ttnn::Tensor::from_vector(words, spec, &mesh); });
    }
    return s;
}

const CombineState& state(
    MoEBlockResources& res,
    uint32_t rows_cap,
    uint32_t m,
    uint32_t activation,
    const fre::FlatLayoutOptions& layout,
    const CombineConfig& combine) {
    // (planned once per resources: a cache hit does a lookup, the hash and the address patches)
    const auto& st = res.plan<CombineState>(
        "routed_experts_combine/state", [&] { return make_state(res, m, activation, layout, combine, rows_cap); });
    TT_FATAL(
        st.combine == combine && st.flat_config.activation == activation &&
            st.flat_config.max_tokens == (m + 31) / 32 * 32 && st.layout == layout && st.rows_cap == rows_cap,
        "routed_experts_combine: the resources' combine was planned for another config");
    return st;
}

}  // namespace routed_experts_combine_detail

uint32_t routed_experts_combine_static_cb_bytes(
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout) {
    using namespace routed_experts_combine_detail;
    const auto [group, head_port] = block_group_and_head_port(resources);
    const auto l1 = plan_routed_experts_combine_l1(
        resources, max_tokens_per_expert, activation, layout, CombineConfig{}, group, head_port);
    return std::max(l1.static_bytes, head_port);
}

tt::tt_metal::TensorSpec routed_experts_combine_arena_spec(
    MoEBlockResources& resources,
    uint32_t rows_cap,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t other_bytes) {
    using namespace routed_experts_combine_detail;
    const auto& st = state(resources, rows_cap, max_tokens_per_expert, activation, layout, combine);
    const auto& sh = *st.arena_spec->memory_config().shard_spec();
    const uint32_t need = sh.shape[0] * 64;
    if (other_bytes <= need) {
        return *st.arena_spec;
    }
    const uint32_t rows = (other_bytes / 64 + 31) / 32 * 32;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({static_cast<uint32_t>(sh.grid.num_cores()) * rows, 32}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig(
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                tt::tt_metal::ShardSpec(sh.grid, {rows, 32}, tt::tt_metal::ShardOrientation::ROW_MAJOR))));
}

ttnn::Tensor routed_experts_combine(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& token_index,
    const ttnn::Tensor& table,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const std::optional<ttnn::Tensor>& arena,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine) {
    using namespace routed_experts_combine_detail;
    const uint32_t rows_cap = table.logical_shape()[-2] * 32;
    const auto& st = state(resources, rows_cap, max_tokens_per_expert, activation, layout, combine);
    auto& mesh = resources.mesh_device();
    // the call's arena: the block's shared one (the fused head ran in it first), else one of our own for the call
    std::optional<ttnn::Tensor> own;
    if (arena) {
        const auto& mc = arena->memory_config();
        const auto& want = *st.arena_spec->memory_config().shard_spec();
        TT_FATAL(
            mc.buffer_type() == BufferType::L1 && mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
                mc.shard_spec().has_value() && mc.shard_spec()->grid.contains(want.grid),
            "routed_experts_combine: the arena must be an L1 tensor height-sharded over every worker core");
        TT_FATAL(
            arena->buffer()->aligned_size_per_bank() >= want.shape[0] * 64,
            "routed_experts_combine: the arena holds {} B a core, the combine needs {} B",
            arena->buffer()->aligned_size_per_bank(),
            want.shape[0] * 64);
    } else {
        own = create_device_tensor(*st.arena_spec, &mesh);
    }
    RoutedExpertsCombineParams params;
    params.combine = combine;
    params.S = st.plan->S;
    params.W = st.plan->W;
    params.flat_config = st.flat_config;
    params.cells = st.cells;
    params.flat = st.handoff;
    params.plan_key = resources.plan_key();
    params.plan = st.plan;
    params.flat_plan = st.flat.plan;
    auto out = ttnn::prim::moe_block_routed_experts_combine(
        params,
        RoutedExpertsCombineInputs{
            .plan = plan,
            .table = table,
            .out = st.out,
            .bar = st.bar,
            .hcache = st.hcache,
            .lut = st.lut,
            .flat = flat_routed_expert::FlatRoutedExpertInputs{
                .x = dispatched,
                .counts = counts,
                .regions = regions,
                .global_expert_ids = global_expert_ids,
                .gate_up_weights = gate_up_weights,
                .down_weights = down_weights,
                .done_words = st.flat.done_words,
                .arena = arena ? *arena : *own,
                .words = st.flat.relay_words,
                .output = *st.y,
                .token_index = token_index}});
    if (own) {
        own->deallocate(true);
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
