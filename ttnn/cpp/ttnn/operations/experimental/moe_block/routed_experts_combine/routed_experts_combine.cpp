// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine.hpp"
#include <tracy/Tracy.hpp>

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
namespace flat_expert = flat_routed_expert;

// What one resources object keeps for the combine across calls: the plans, the persistent tensors, the arena's spec.
struct CombineState {
    flat_expert::FlatRoutedExpertConfig flat_config;
    flat_expert::FlatLayoutOptions layout;  // (as the caller passed it)
    flat_expert::FlatRoutedExpertResources flat;
    CombineConfig combine;
    std::vector<CoreCoord> free_cores;
    FlatHandoff handoff;
    std::shared_ptr<const CombinePlan> plan;
    ttnn::Tensor out, barrier, hcache;
    std::optional<ttnn::Tensor> accumulator_lut;
    std::optional<tt::tt_metal::TensorSpec> arena_spec;
    std::optional<ttnn::Tensor> y;  // the flat expert's y (never written with the L1 hand-off)
    uint32_t rows_cap = 0;
};

tt::tt_metal::TensorSpec rm_spec(const ttnn::Shape& shape, DataType data_type, const MemoryConfig& memory_config) {
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(data_type, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

flat_expert::FlatRoutedExpertConfig base_flat_config(
    MoEBlockResources& resources, uint32_t m, const flat_expert::FlatLayoutOptions& layout) {
    const auto& dims = resources.dims();
    const uint32_t N = resources.topology().num_chips();
    TT_FATAL(dims.num_experts % N == 0, "routed_experts_combine: {} experts over {} chips", dims.num_experts, N);
    auto flat_config = flat_expert::make_flat_routed_expert_config(
        dims.routed_width(),
        dims.routed_expert_hidden_dim,
        dims.num_experts / N,
        dims.num_experts,
        m,
        resources.dtypes().routed_weights == DataType::BFLOAT8_B,
        1,
        layout);
    flat_config.activation =
        routed_activation_code(dims.routed_expert_activation);  // (the model's, from the resources)
    flat_config.down_fp32 = true;
    return flat_config;
}

// L1 a core the block's own programs that run while its arena is allocated take for static circular buffers: the
// flat program's only (the head's and the dispatch's CBs are in the arena - MoEBlock passes it to both; the row group
// sizes the dispatch's arena region, not a static CB).
uint32_t static_cb_bytes(
    MoEBlockResources& /*resources*/, const flat_expert::FlatRoutedExpertPlan& flat_plan, uint32_t /*group*/) {
    return align64(flat_plan.static_l1_bytes);
}

// The cores the flat plan leaves free, row-major.
std::vector<CoreCoord> free_cores_of(MoEBlockResources& resources, const flat_expert::FlatRoutedExpertPlan& flat_plan) {
    const auto grid = resources.mesh_device().compute_with_storage_grid_size();
    const auto taken = flat_plan.arena_cores();
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
std::pair<uint32_t, uint32_t> block_group_and_head_port(const MoEBlockResources& resources) {
    const auto l1 = block_l1_plan(resources);
    return {sparse_dispatch_2d::dispatch_group(resources), l1 ? l1->head_port_cb : 0u};
}

}  // namespace routed_experts_combine_detail

CombineL1Plan plan_routed_experts_combine_l1(
    MoEBlockResources& resources,
    uint32_t m,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t group,
    uint32_t head_port_cb) {
    using namespace routed_experts_combine_detail;
    auto& mesh = resources.mesh_device();
    const auto& block_config = resources.config();
    const uint32_t N = resources.topology().num_chips();
    TT_FATAL(
        block_config.split_tokens() % N == 0,
        "routed_experts_combine: chunk {} over {} chips",
        block_config.split_tokens(),
        N);
    // The flat expert as the caller gave it when its plan fits the arena the combine gets, else with that arena as
    // its budget (FlatLayoutOptions.arena_bytes: the flat plan fits it) - a smaller plan leaves the arena at least as
    // large, so the fitted plan still fits.
    CombineL1Plan out;
    out.flat = base_flat_config(resources, m, layout);
    out.arena_bytes = combine.arena_bytes;
    std::shared_ptr<const flat_routed_expert::FlatRoutedExpertPlan> flat_plan;
    for (uint32_t i = 0; i < 4; ++i) {
        flat_plan = flat_expert::flat_routed_expert_plan(mesh, out.flat);
        out.static_bytes = static_cb_bytes(resources, *flat_plan, group);
        if (combine.arena_bytes == 0) {  // (the largest free L1 block less what stays below the arena)
            const uint32_t reserve = exchange_arena_reserve(std::max(out.static_bytes, head_port_cb));
            const auto stats = mesh.allocator()->get_statistics(tt::tt_metal::BufferType::L1);
            out.arena_bytes = exchange_arena_bytes(static_cast<uint32_t>(stats.largest_free_block_bytes), reserve);
        }
        if (flat_plan->arena_tiles * 2048 <= out.arena_bytes || layout.arena_bytes > 0) {
            break;  // (a caller's own budget is kept: checked below)
        }
        out.flat.layout.arena_bytes = static_cast<int32_t>(out.arena_bytes);
    }
    TT_FATAL(
        out.arena_bytes >= flat_plan->arena_tiles * 2048,
        "routed_experts_combine: the arena {} KB < the flat expert's {} KB",
        out.arena_bytes >> 10,
        (flat_plan->arena_tiles * 2048) >> 10);
    out.free_cores = free_cores_of(resources, *flat_plan);
    CombinePlanInputs plan_inputs;
    plan_inputs.home_tokens = block_config.split_tokens() / N;
    plan_inputs.row_elements = resources.dims().routed_width();
    plan_inputs.free_cores = out.free_cores;
    plan_inputs.accumulator_count = combine.accumulator_count;
    plan_inputs.mode = combine.mode;
    plan_inputs.depth = combine.depth;
    plan_inputs.landing = combine.landing;
    plan_inputs.credit_batch = combine.credit_batch;
    plan_inputs.arena_bytes = out.arena_bytes;
    if (!combine.landing || landing_on_flat(*combine.landing)) {  // (the flat expert cores' free L1)
        for (auto& f : flat_plan->free_l1(out.arena_bytes)) {
            plan_inputs.flat_free.push_back({std::move(f.cores), std::move(f.spans)});
        }
    }
    out.plan = make_combine_plan(mesh, resources, plan_inputs);
    return out;
}

namespace routed_experts_combine_detail {

CombineState make_state(
    MoEBlockResources& resources,
    uint32_t m,
    const flat_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t rows_cap) {
    auto& mesh = resources.mesh_device();
    const uint32_t N = resources.topology().num_chips();
    CombineState s;
    s.combine = combine;
    s.layout = layout;
    // (in the block: its L1 plan's dispatch group and the head's ports below the shared arena)
    const auto [group, head_port] = block_group_and_head_port(resources);
    const CombineL1Plan l1 = plan_routed_experts_combine_l1(resources, m, layout, combine, group, head_port);
    s.flat_config = l1.flat;
    s.flat = flat_expert::ensure_flat_routed_expert_resources(resources, s.flat_config);
    s.flat_config = flat_expert::with_combine_overlap(s.flat_config, *s.flat.plan, true);
    s.free_cores = l1.free_cores;
    s.handoff.writers = flat_expert::flat_y_writers(*s.flat.plan);
    s.plan = l1.plan;
    {  // (the program's flat plan leaves the same L1 free as the one the pool was carved from)
        const auto program_flat_plan = flat_expert::flat_routed_expert_plan(mesh, s.flat_config);
        const auto& pool_flat_plan = *flat_expert::flat_routed_expert_plan(mesh, l1.flat);
        TT_FATAL(
            !s.plan->landing_on_flat_expert_cores ||
                (program_flat_plan->schedule_page_offset == pool_flat_plan.schedule_page_offset &&
                 program_flat_plan->dynamic_counts_offset == pool_flat_plan.dynamic_counts_offset &&
                 program_flat_plan->gate_up_arena_bytes == pool_flat_plan.gate_up_arena_bytes &&
                 program_flat_plan->relay_landing_offset == pool_flat_plan.relay_landing_offset &&
                 program_flat_plan->relay_arena_bytes == pool_flat_plan.relay_arena_bytes &&
                 program_flat_plan->arena_tiles == pool_flat_plan.arena_tiles),
            "routed_experts_combine: the flat expert's arena differs from the one the landing pool was planned on");
    }
    const uint32_t S = s.plan->home_tokens, W = s.plan->row_elements;

    // the call's arena: every worker core, the combine's bytes, the writers' words, the kernels' args blocks (64 B
    // rows)
    const auto grid = mesh.compute_with_storage_grid_size();
    const uint32_t rows = l1.arena_bytes / 64 + kWriterWordsRows + kArgsRows;  // (+ the kernels' args blocks)
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
    s.barrier = resources.persistent_words("routed_experts_combine/bar", all, 16);
    s.hcache = resources.persistent_tensor(
        "routed_experts_combine/hcache",
        rm_spec(ttnn::Shape({8, hcache_page_bytes(N) / 4}), DataType::UINT32, DRAM_MEMORY_CONFIG));
    s.out = resources.persistent_tensor(
        "routed_experts_combine/out", rm_spec(ttnn::Shape({1, 1, S, W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
    s.rows_cap = rows_cap;
    s.y = resources.persistent_tensor(
        "routed_experts_combine/y", rm_spec(ttnn::Shape({rows_cap, W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
    if (s.plan->per_chip_accumulator_table) {
        const auto& lut16 = s.plan->accumulator_table;
        std::vector<uint32_t> words((lut16.size() + 1) / 2, 0);
        for (size_t i = 0; i < lut16.size(); ++i) {
            words[i / 2] |= static_cast<uint32_t>(lut16[i]) << (16 * (i % 2));
        }
        const auto spec =
            rm_spec(ttnn::Shape({1, static_cast<uint32_t>(words.size())}), DataType::UINT32, DRAM_MEMORY_CONFIG);
        s.accumulator_lut = resources.plan<ttnn::Tensor>(
            "routed_experts_combine/lut", [&] { return ttnn::Tensor::from_vector(words, spec, &mesh); });
    }
    return s;
}

const CombineState& state(
    MoEBlockResources& resources,
    uint32_t rows_cap,
    uint32_t m,
    const flat_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine) {
    // (planned once per resources: a cache hit does a lookup, the hash and the address patches)
    const auto& combine_state = resources.plan<CombineState>(
        "routed_experts_combine/state", [&] { return make_state(resources, m, layout, combine, rows_cap); });
    TT_FATAL(
        combine_state.combine == combine && combine_state.flat_config.max_tokens == (m + 31) / 32 * 32 &&
            combine_state.layout == layout && combine_state.rows_cap == rows_cap,
        "routed_experts_combine: the resources' combine was planned for another config");
    return combine_state;
}

}  // namespace routed_experts_combine_detail

tt::tt_metal::TensorSpec routed_experts_combine_arena_spec(
    MoEBlockResources& resources,
    uint32_t rows_cap,
    uint32_t max_tokens_per_expert,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t other_bytes) {
    using namespace routed_experts_combine_detail;
    const auto& combine_state = state(resources, rows_cap, max_tokens_per_expert, layout, combine);
    const auto& shard_spec = *combine_state.arena_spec->memory_config().shard_spec();
    const uint32_t need = shard_spec.shape[0] * 64;
    if (other_bytes <= need) {
        return *combine_state.arena_spec;
    }
    const uint32_t rows = (other_bytes / 64 + 31) / 32 * 32;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({static_cast<uint32_t>(shard_spec.grid.num_cores()) * rows, 32}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig(
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                tt::tt_metal::ShardSpec(shard_spec.grid, {rows, 32}, tt::tt_metal::ShardOrientation::ROW_MAJOR))));
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
    const std::optional<ttnn::Tensor>& arena,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine) {
    ZoneScopedN("routed_experts_combine: entry");
    using namespace routed_experts_combine_detail;
    const uint32_t rows_cap = table.logical_shape()[-2] * 32;
    const auto& combine_state = state(resources, rows_cap, max_tokens_per_expert, layout, combine);
    auto& mesh = resources.mesh_device();
    // the call's arena: the block's shared one (the MoE head ran in it first), else one of our own for the call
    std::optional<ttnn::Tensor> own;
    if (arena) {
        const auto& arena_memory_config = arena->memory_config();
        const auto& want = *combine_state.arena_spec->memory_config().shard_spec();
        TT_FATAL(
            arena_memory_config.buffer_type() == BufferType::L1 &&
                arena_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
                arena_memory_config.shard_spec().has_value() &&
                arena_memory_config.shard_spec()->grid.contains(want.grid),
            "routed_experts_combine: the arena must be an L1 tensor height-sharded over every worker core");
        TT_FATAL(
            arena->buffer()->aligned_size_per_bank() >= want.shape[0] * 64,
            "routed_experts_combine: the arena holds {} B a core, the combine needs {} B",
            arena->buffer()->aligned_size_per_bank(),
            want.shape[0] * 64);
    } else {
        own = create_device_tensor(*combine_state.arena_spec, &mesh);
    }
    RoutedExpertsCombineParams params;
    params.combine = combine;
    params.S = combine_state.plan->home_tokens;
    params.W = combine_state.plan->row_elements;
    params.flat_config = combine_state.flat_config;
    params.free_cores = combine_state.free_cores;
    params.flat = combine_state.handoff;
    params.plan_key = resources.plan_key();
    params.plan = combine_state.plan;
    params.flat_plan = combine_state.flat.plan;
    auto out = ttnn::prim::moe_block_routed_experts_combine(
        params,
        RoutedExpertsCombineInputs{
            .plan = plan,
            .table = table,
            .out = combine_state.out,
            .barrier = combine_state.barrier,
            .hcache = combine_state.hcache,
            .accumulator_lut = combine_state.accumulator_lut,
            .flat = flat_routed_expert::FlatRoutedExpertInputs{
                .x = dispatched,
                .counts = counts,
                .regions = regions,
                .global_expert_ids = global_expert_ids,
                .gate_up_weights = gate_up_weights,
                .down_weights = down_weights,
                .done_words = combine_state.flat.done_words,
                .arena = arena ? *arena : *own,
                .words = combine_state.flat.relay_words,
                .output = *combine_state.y,
                .token_index = token_index}});
    if (own) {
        own->deallocate(true);
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
