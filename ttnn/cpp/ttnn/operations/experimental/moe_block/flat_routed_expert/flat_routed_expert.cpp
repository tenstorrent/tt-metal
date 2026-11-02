// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

#include <tt-metalium/mesh_device.hpp>

#include "device/flat_routed_expert_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace host_detail {
using tt::tt_metal::CoreCoord;

// [cores x rows, 32] bf16 TILE, height-sharded `rows` a core over the cores' rectangles
tt::tt_metal::TensorSpec l1_sharded_spec(const std::vector<CoreCoord>& cores, uint32_t rows) {
    using namespace tt::tt_metal;
    const MemoryConfig mc(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(rect_ranges(cores), {rows, 32}, ShardOrientation::ROW_MAJOR));
    return TensorSpec(
        ttnn::Shape({static_cast<uint32_t>(cores.size()) * rows, 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mc));
}

// The per-launch scratch specs (arena, relay words) of a plan, built once: rect_ranges + the sharded spec cost tens
// of us per call otherwise. Plans live in the never-freed plan cache, so their address is a stable key.
const std::pair<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>& scratch_specs(const FlatRoutedExpertPlan& plan) {
    static std::mutex mu;
    static std::map<const FlatRoutedExpertPlan*, std::pair<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(&plan);
    if (it == cache.end()) {
        it = cache
                 .emplace(
                     &plan,
                     std::make_pair(
                         l1_sharded_spec(plan.arena_cores(), plan.arena_tiles * 32), l1_sharded_spec(plan.relays, 32)))
                 .first;
    }
    return it->second;
}

// words a down coordinator's done-word shard holds: one a down core, 16 at least
uint32_t done_words_per_core(const FlatRoutedExpertPlan& plan) {
    return std::max<uint32_t>(16, (static_cast<uint32_t>(plan.down.size()) + 15) / 16 * 16);
}
}  // namespace host_detail

FlatRoutedExpertConfig make_flat_routed_expert_config(
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t experts_per_chip,
    uint32_t num_global_experts,
    uint32_t max_tokens_per_expert,
    bool weights_bf8,
    uint32_t pin,
    const FlatLayoutOptions& layout) {
    return FlatRoutedExpertConfig{
        .hidden = hidden,
        .intermediate = intermediate,
        .experts_per_chip = experts_per_chip,
        .num_global_experts = num_global_experts,
        .max_tokens = (max_tokens_per_expert + 31) / 32 * 32,
        .weights_bf8 = weights_bf8,
        .pin = pin,
        .layout = layout};
}

ttnn::Tensor create_flat_routed_expert_done_words(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertPlan& plan,
    tt::tt_metal::BufferType buffer_type) {
    using namespace tt::tt_metal;
    const uint32_t nw = host_detail::done_words_per_core(plan);
    const MemoryConfig mc(
        TensorMemoryLayout::HEIGHT_SHARDED,
        buffer_type,
        ShardSpec(rect_ranges(plan.coords), {1, nw}, ShardOrientation::ROW_MAJOR));
    return ttnn::zeros(
        ttnn::Shape({static_cast<uint32_t>(plan.coords.size()), nw}),
        DataType::UINT32,
        Layout::ROW_MAJOR,
        std::ref(mesh_device),
        mc);
}

ttnn::Tensor flat_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const ttnn::Tensor& done_words,
    const FlatDriscSenders& drisc_senders,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    uint32_t pin,
    const std::optional<ttnn::Tensor>& token_index,
    uint32_t x_pages_per_row,
    bool y_row_major,
    bool down_fp32,
    bool pack_stochastic_rounding,
    bool y_bf16,
    uint32_t math_fidelity,
    const FlatLayoutOptions& layout,
    const std::optional<ttnn::Tensor>& output_tensor) {
    using namespace tt::tt_metal;
    TT_FATAL(!(y_bf16 && y_row_major), "flat_routed_expert: y_bf16 is for tiled y (row-major y is bf16 already)");
    auto* device = x.device();
    TT_FATAL(
        drisc_senders.running() && drisc_senders.mesh_device() == device,
        "flat_routed_expert: the DRISC senders must be running on this mesh device");
    FlatRoutedExpertConfig cfg = make_flat_routed_expert_config(
        x.logical_shape()[-1] * x_pages_per_row,
        intermediate,
        static_cast<uint32_t>(global_expert_ids.logical_volume()),
        counts.logical_shape()[-1],
        max_tokens_per_expert,
        gate_up_weights.dtype() == DataType::BFLOAT8_B,
        pin,
        layout);
    cfg.activation = activation;
    cfg.y_row_major = y_row_major;
    cfg.down_fp32 = down_fp32;
    cfg.pack_stochastic_rounding = pack_stochastic_rounding;
    cfg.y_bf16 = y_bf16;
    cfg.math_fidelity = math_fidelity;
    const auto plan = flat_routed_expert_plan(*device, cfg);
    TT_FATAL(
        drisc_senders.n_slots() == plan->drisc_n && drisc_senders.tiles() == plan->drisc_tiles,
        "flat_routed_expert: DRISC senders of {} slots ({}), the plan has {} ({})",
        drisc_senders.n_slots(),
        to_string(drisc_senders.tiles()),
        plan->drisc_n,
        to_string(plan->drisc_tiles));
    // (output: a caller's tensor, e.g. a persistent buffer another program reads at a fixed address; validated)
    const ttnn::Tensor output =
        output_tensor
            ? *output_tensor
            : ttnn::empty(
                  ttnn::Shape(
                      {token_index ? token_index->logical_shape()[-1] : x.logical_shape()[-2] / x_pages_per_row,
                       cfg.hidden}),
                  y_row_major || y_bf16 ? DataType::BFLOAT16 : DataType::BFLOAT8_B,
                  y_row_major ? Layout::ROW_MAJOR : Layout::TILE,
                  device,
                  MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    // per-launch scratch: the arena (every role's buffers) and the relays' freed words; freed right after the launch
    // (the device reads them in order), so they never pin L1 against the next op's circular buffers
    const auto& [arena_spec, words_spec] = host_detail::scratch_specs(*plan);
    ttnn::Tensor arena = create_device_tensor(arena_spec, device);
    ttnn::Tensor words = create_device_tensor(words_spec, device);
    auto y = ttnn::prim::moe_block_flat_routed_expert(
        cfg,
        FlatRoutedExpertInputs{
            .x = x,
            .counts = counts,
            .regions = regions,
            .global_expert_ids = global_expert_ids,
            .gate_up_weights = gate_up_weights,
            .down_weights = down_weights,
            .done_words = done_words,
            .arena = arena,
            .words = words,
            .output = output,
            .token_index = token_index});
    arena.deallocate(true);
    words.deallocate(true);
    return y;
}

FlatRoutedExpertResources ensure_flat_routed_expert_resources(
    MoEBlockResources& resources, const FlatRoutedExpertConfig& config) {
    auto& mesh_device = resources.mesh_device();
    FlatRoutedExpertResources r;
    r.plan = flat_routed_expert_plan(mesh_device, config);
    r.done_words = resources.persistent_words(
        "flat_routed_expert/done", rect_ranges(r.plan->coords), host_detail::done_words_per_core(*r.plan));
    // (the relays' words hold the gate/up cores' freed words and the x-ready word at X_READY_OFF)
    r.relay_words =
        resources.persistent_words("flat_routed_expert/relay_words", rect_ranges(r.plan->relays), X_READY_OFF / 4 + 1);
    TT_FATAL(
        4 * r.plan->gu.size() <= X_READY_OFF,
        "flat_routed_expert: {} gate/up cores' freed words overrun the x-ready word at {}",
        r.plan->gu.size(),
        X_READY_OFF);
    // (one attachment per tile choice: a request for another choice relaunches the mesh's senders,
    // FlatDriscSenders::get)
    const std::string drisc_name = r.plan->drisc_tiles == FlatDriscTiles::both
                                       ? std::string("flat_routed_expert/drisc")
                                       : std::string("flat_routed_expert/drisc/") + to_string(r.plan->drisc_tiles);
    r.drisc_senders = FlatDriscSenders::get(mesh_device, r.plan->drisc_n, r.plan->drisc_tiles);
    if (!resources.has(drisc_name)) {
        resources.attach(drisc_name, r.drisc_senders);
    }
    return r;
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
