// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine.hpp"

#include <set>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "device/routed_experts_combine_device_operation.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_program.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_detail {

using tt::tt_metal::CoreCoord;
namespace fre = flat_routed_expert;

// What one resources object keeps for the combine across calls: the plans, the persistent tensors, the arena's spec.
struct CombineState {
    fre::FlatRoutedExpertConfig flat_config;
    fre::FlatRoutedExpertResources flat;
    CombineConfig combine;
    std::vector<CoreCoord> cells;
    FlatHandoff handoff;
    std::shared_ptr<const CombinePlan> plan;
    ttnn::Tensor out, bar, hcache;
    std::optional<ttnn::Tensor> lut, sems;
    std::optional<tt::tt_metal::TensorSpec> arena_spec, y_spec;
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
    auto lay = layout;
    if (res.config().experts_during_dispatch) {
        lay.x_ready = 8;  // (the dispatch ports a chip at most: each chip's count is padded to it)
    }
    auto cfg = fre::make_flat_routed_expert_config(
        dims.routed_width(),
        dims.routed_expert_hidden_dim,
        dims.num_experts / N,
        dims.num_experts,
        m,
        res.dtypes().routed_weights == DataType::BFLOAT8_B,
        1,
        lay);
    cfg.activation = activation;
    cfg.down_fp32 = true;
    return cfg;
}

CombineState make_state(
    MoEBlockResources& res,
    uint32_t m,
    uint32_t activation,
    const fre::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t rows_cap) {
    auto& mesh = res.mesh_device();
    const auto& cfg = res.config();
    const auto& topo = res.topology();
    const uint32_t N = topo.num_chips();
    TT_FATAL(cfg.chunk_tokens % N == 0, "routed_experts_combine: chunk {} over {} chips", cfg.chunk_tokens, N);
    CombineState s;
    s.combine = combine;
    s.flat_config = base_flat_config(res, m, activation, layout);
    s.flat = fre::ensure_flat_routed_expert_resources(res, s.flat_config);
    s.flat_config = fre::with_combine_overlap(s.flat_config, *s.flat.plan, true);
    const auto& fp = *s.flat.plan;
    // the cells the flat plan leaves (k3 combine_cells), its y writers
    const auto grid = mesh.compute_with_storage_grid_size();
    const auto taken = fp.arena_cores();
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (std::none_of(taken.begin(), taken.end(), [&](const CoreCoord& c) { return c.x == x && c.y == y; })) {
                s.cells.emplace_back(x, y);
            }
        }
    }
    s.handoff.writers = fre::flat_y_writers(fp);
    s.handoff.relays = fp.relays;

    CombinePlanInputs in;
    in.S = cfg.chunk_tokens / N;
    in.W = res.dims().routed_width();
    in.cells = s.cells;
    in.n_acc = combine.n_acc;
    in.mode = combine.mode;
    if (cfg.experts_during_dispatch && !in.mode) {
        in.mode = CombineMode::PortsApart;  // (the dispatch runs on the combine's ports: one a (direction, link))
    }
    in.depth = combine.depth;
    in.piece_max = combine.piece_max;
    in.arena_bytes = combine.arena_bytes;
    if (in.arena_bytes == 0) {  // (the reference's sizing: the largest free L1 block less what stays below it)
        const uint32_t group = 2 * in.W <= 7168 ? 8 : 2;
        const auto ml = manifest_layout(topo.R, topo.C, topo.wrap_x, topo.wrap_y, in.S);
        const uint32_t reserve = exchange_arena_reserve(
            fp.static_l1_bytes,
            dispatch_static_cb_bytes(N, in.S, in.W, group),
            align64(ml.total_entries * 16ull),
            cfg.experts_during_dispatch);
        const auto stats = mesh.allocator()->get_statistics(tt::tt_metal::BufferType::L1);
        in.arena_bytes = exchange_arena_bytes(static_cast<uint32_t>(stats.largest_free_block_bytes), reserve);
    }
    TT_FATAL(
        in.arena_bytes >= fp.arena_tiles * 2048,
        "routed_experts_combine: the arena {} KB < the flat expert's {} KB",
        in.arena_bytes >> 10,
        (fp.arena_tiles * 2048) >> 10);
    s.plan = make_combine_plan(mesh, res, in);

    // the call's arena: every worker core, the combine's bytes then the writers' words (rows of 64 B)
    const uint32_t rows = in.arena_bytes / 64 + kWriterWordsRows;
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
        "routed_experts_combine/out", rm_spec(ttnn::Shape({1, 1, in.S, in.W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
    // (the flat expert's y: with the L1 hand-off never written, its spec only)
    s.y_spec = rm_spec(ttnn::Shape({rows_cap, in.W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG);
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
    if (s.plan->dispatch) {  // (the dispatch phase's semaphore words: 3 ring counters + the fence's ready, 16 B apart)
        std::set<std::pair<uint32_t, uint32_t>> cells;
        for (const auto& [k, c] : s.plan->port_core) {
            cells.emplace(c.y, c.x);
        }
        std::vector<CoreRange> pr;
        for (const auto& [y, x] : cells) {
            pr.emplace_back(CoreCoord(x, y), CoreCoord(x, y));
        }
        s.sems = res.persistent_words("routed_experts_combine/dispatch_sems", CoreRangeSet(pr), 16);
    }
    return s;
}

}  // namespace routed_experts_combine_detail

flat_routed_expert::FlatRoutedExpertConfig routed_experts_combine_flat_config(
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout) {
    using namespace routed_experts_combine_detail;
    auto cfg = base_flat_config(resources, max_tokens_per_expert, activation, layout);
    const auto fr = fre::ensure_flat_routed_expert_resources(resources, cfg);
    return fre::with_combine_overlap(cfg, *fr.plan, true);
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
    const std::optional<ttnn::Tensor>& route,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine) {
    using namespace routed_experts_combine_detail;
    const uint32_t rows_cap = table.logical_shape()[-2] * 32;
    // (planned once per resources: a cache hit does a lookup, the arena's allocation, the hash and the patches)
    const auto& st = resources.plan<CombineState>("routed_experts_combine/state", [&] {
        return make_state(resources, max_tokens_per_expert, activation, layout, combine, rows_cap);
    });
    TT_FATAL(
        st.combine == combine && st.flat_config.activation == activation &&
            st.flat_config.max_tokens == (max_tokens_per_expert + 31) / 32 * 32,
        "routed_experts_combine: the resources' combine was planned for another config");
    auto& mesh = resources.mesh_device();
    ttnn::Tensor arena = create_device_tensor(*st.arena_spec, &mesh);
    const auto y = resources.persistent_tensor("routed_experts_combine/y", *st.y_spec);
    RoutedExpertsCombineParams params;
    params.combine = combine;
    params.S = st.plan->S;
    params.W = st.plan->W;
    params.dispatch_in_program = resources.config().experts_during_dispatch;
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
            .route = route,
            .sems = route ? st.sems : std::nullopt,
            .flat = flat_routed_expert::FlatRoutedExpertInputs{
                .x = dispatched,
                .counts = counts,
                .regions = regions,
                .global_expert_ids = global_expert_ids,
                .gate_up_weights = gate_up_weights,
                .down_weights = down_weights,
                .done_words = st.flat.done_words,
                .arena = arena,
                .words = st.flat.relay_words,
                .output = y,
                .token_index = token_index}});
    arena.deallocate(true);
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
