// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "device/routed_experts_combine_device_operation.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_detail {

using tt::tt_metal::CoreCoord;

// What one resources object keeps for the combine across calls (the plan and its persistent tensors).
struct CombineState {
    CombinePlanInputs in;
    CombineConfig combine;
    std::vector<CoreCoord> writers;
    uint32_t words_offset = 0;
    std::shared_ptr<const CombinePlan> plan;
    ttnn::Tensor out, bar, hcache;
    std::optional<ttnn::Tensor> lut;
};

tt::tt_metal::TensorSpec rm_spec(const ttnn::Shape& shape, DataType dt, const MemoryConfig& mc) {
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), mc));
}

CombineState make_state(
    MoEBlockResources& res,
    const std::vector<CoreCoord>& cells,
    const std::vector<CoreCoord>& writers,
    uint32_t words_offset,
    const CombineConfig& combine) {
    auto& mesh = res.mesh_device();
    const auto& cfg = res.config();
    const auto& topo = res.topology();
    const uint32_t N = topo.num_chips();
    TT_FATAL(cfg.chunk_tokens % N == 0, "routed_experts_combine: chunk {} over {} chips", cfg.chunk_tokens, N);
    CombineState s;
    s.in.S = cfg.chunk_tokens / N;
    s.in.W = res.dims().routed_width();
    s.in.cells = cells;
    s.in.n_acc = combine.n_acc;
    s.in.mode = combine.mode;
    s.in.depth = combine.depth;
    s.in.piece_max = combine.piece_max;
    s.in.arena_bytes = combine.arena_bytes;
    if (s.in.arena_bytes == 0) {  // (the reference's sizing: the largest free L1 block less what stays below it)
        const uint32_t S = s.in.S, W = s.in.W;
        const uint32_t group = 2 * W <= 7168 ? 8 : 2;
        const auto ml = manifest_layout(topo.R, topo.C, topo.wrap_x, topo.wrap_y, S);
        const uint32_t reserve = exchange_arena_reserve(
            combine.flat_static_bytes,
            dispatch_static_cb_bytes(N, S, W, group),
            align64(ml.total_entries * 16ull),
            cfg.experts_during_dispatch);
        const auto stats = mesh.allocator()->get_statistics(tt::tt_metal::BufferType::L1);
        s.in.arena_bytes = exchange_arena_bytes(static_cast<uint32_t>(stats.largest_free_block_bytes), reserve);
    }
    s.combine = combine;
    s.writers = writers;
    s.words_offset = words_offset;
    s.plan = make_combine_plan(mesh, res, s.in);
    const auto grid = mesh.compute_with_storage_grid_size();
    s.bar = res.persistent_words(
        "routed_experts_combine/bar", CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), 16);
    s.hcache = res.persistent_tensor(
        "routed_experts_combine/hcache",
        rm_spec(ttnn::Shape({8, hcache_page_bytes(N) / 4}), DataType::UINT32, DRAM_MEMORY_CONFIG));
    s.out = res.persistent_tensor(
        "routed_experts_combine/out",
        rm_spec(ttnn::Shape({1, 1, s.in.S, s.in.W}), DataType::BFLOAT16, DRAM_MEMORY_CONFIG));
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

}  // namespace routed_experts_combine_detail

ttnn::Tensor routed_experts_combine(
    const ttnn::Tensor& plan,
    const ttnn::Tensor& table,
    MoEBlockResources& resources,
    const std::vector<tt::tt_metal::CoreCoord>& cells,
    const std::vector<tt::tt_metal::CoreCoord>& writers,
    uint32_t writer_words_offset,
    const CombineConfig& combine) {
    using namespace routed_experts_combine_detail;
    // (planned once per resources: a cache hit does a lookup, the hash and the address patches only)
    const auto& st = resources.plan<CombineState>("routed_experts_combine/state", [&] {
        return make_state(resources, cells, writers, writer_words_offset, combine);
    });
    TT_FATAL(
        st.in.cells == cells && st.combine == combine && st.writers == writers &&
            st.words_offset == writer_words_offset,
        "routed_experts_combine: the resources' combine was planned for other cells / writers / config");
    RoutedExpertsCombineParams params;
    params.combine = combine;
    params.S = st.in.S;
    params.W = st.in.W;
    params.dispatch_in_program = resources.config().experts_during_dispatch;
    params.cells = cells;
    params.flat = FlatHandoff{writers, writer_words_offset};
    params.plan_key = resources.plan_key();
    params.plan = st.plan;
    return ttnn::prim::moe_block_routed_experts_combine(
        params, RoutedExpertsCombineInputs{plan, table, st.out, st.bar, st.hcache, st.lut});
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
