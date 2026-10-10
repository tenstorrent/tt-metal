// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert.hpp"

#include <map>
#include <mutex>
#include <tuple>

#include "device/flat_routed_expert_device_operation.hpp"
#include <set>

#include "device/flat_combine_overlap_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_placement.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& c) {
    using Key = std::tuple<const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, bool, bool>;
    static std::mutex mu;
    static std::map<Key, std::shared_ptr<const FlatRoutedExpertPlan>> cache;
    const Key key{
        device,
        c.hidden,
        c.intermediate,
        c.experts_per_chip,
        c.num_global_experts,
        c.max_tokens,
        c.weights_bf8,
        c.pin,
        c.x_bf16,
        c.h_bf16};
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, std::make_shared<const FlatRoutedExpertPlan>(make_flat_routed_expert_plan(device, c)))
                 .first;
    }
    return it->second;
}

namespace {
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

// The per-launch scratch specs (arena, relay words) of a plan, built once: rect_ranges + the sharded spec cost ~tens
// of us per call otherwise.
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
}  // namespace

ttnn::Tensor flat_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const std::optional<ttnn::Tensor>& reader_down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    uint32_t pin,
    const std::optional<ttnn::Tensor>& token_index,
    uint32_t x_pages_per_row,
    bool y_row_major,
    bool down_fp32,
    bool pack_stochastic_rounding,
    bool x_bf16,
    bool h_bf16) {
    using namespace tt::tt_metal;
    FlatRoutedExpertConfig cfg{
        .hidden = x.logical_shape()[-1] * x_pages_per_row,
        .intermediate = intermediate,
        .experts_per_chip = static_cast<uint32_t>(global_expert_ids.logical_volume()),
        .num_global_experts = counts.logical_shape()[-1],
        .max_tokens = (max_tokens_per_expert + 31) / 32 * 32,
        .weights_bf8 = gate_up_weights.dtype() == DataType::BFLOAT8_B,
        .activation = activation,
        .pin = pin,
        .y_row_major = y_row_major,
        .down_fp32 = down_fp32,
        .pack_stochastic_rounding = pack_stochastic_rounding,
        .x_bf16 = x_bf16,
        .h_bf16 = h_bf16};
    auto* device = x.device();
    const auto plan = flat_routed_expert_plan(device, cfg);
    TT_FATAL(
        plan->rdown == reader_down_weights.has_value(),
        "flat_routed_expert: reader_down_weights iff the plan has reader tails");
    const ttnn::Tensor output = ttnn::empty(
        ttnn::Shape(
            {token_index ? token_index->logical_shape()[-1] : x.logical_shape()[-2] / x_pages_per_row, cfg.hidden}),
        y_row_major ? DataType::BFLOAT16 : DataType::BFLOAT8_B,
        y_row_major ? Layout::ROW_MAJOR : Layout::TILE,
        device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    // per-launch scratch: the arena (every role's buffers) and the relays' freed words; freed right after the launch
    // (the device reads them in order), so they never pin L1 against the next op's circular buffers
    const auto& [arena_spec, words_spec] = scratch_specs(*plan);  // plans live in the (never freed) plan cache
    ttnn::Tensor arena = create_device_tensor(arena_spec, device);
    ttnn::Tensor words = create_device_tensor(words_spec, device);
    auto y = ttnn::prim::bringup::flat_routed_expert(
        cfg,
        FlatRoutedExpertInputs{
            .x = x,
            .counts = counts,
            .regions = regions,
            .global_expert_ids = global_expert_ids,
            .gate_up_weights = gate_up_weights,
            .down_weights = down_weights,
            .reader_down_weights = reader_down_weights,
            .done_words = done_words,
            .arena = arena,
            .words = words,
            .output = output,
            .token_index = token_index});
    arena.deallocate(true);
    words.deallocate(true);
    return y;
}

ttnn::Tensor flat_combine_overlap(
    const ttnn::Tensor& x,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const std::optional<ttnn::Tensor>& reader_down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_offsets,
    const ttnn::Tensor& replicated_global_expert_idx_table,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t combine_axis,
    uint32_t combine_num_links,
    const tt::tt_metal::GlobalSemaphore& fwd_arrived_semaphore,
    const tt::tt_metal::GlobalSemaphore& final_arrived_semaphore,
    const tt::tt_metal::GlobalSemaphore& expert_go_semaphore,
    uint32_t activation,
    uint32_t pin,
    bool down_fp32,
    bool x_bf16,
    bool h_bf16,
    bool y_row_major,
    const std::optional<ttnn::Tensor>& y_out) {
    using namespace tt::tt_metal;
    FlatRoutedExpertConfig cfg{
        .hidden = x.logical_shape()[-1],
        .intermediate = intermediate,
        .experts_per_chip = static_cast<uint32_t>(global_expert_ids.logical_volume()),
        .num_global_experts = counts.logical_shape()[-1],
        .max_tokens = (max_tokens_per_expert + 31) / 32 * 32,
        .weights_bf8 = gate_up_weights.dtype() == DataType::BFLOAT8_B,
        .activation = activation,
        .pin = pin,
        .y_row_major = y_row_major,
        .down_fp32 = down_fp32,
        .pack_stochastic_rounding = false,
        .x_bf16 = x_bf16,
        .h_bf16 = h_bf16};
    auto* device = x.device();
    const auto plan = flat_routed_expert_plan(device, cfg);
    TT_FATAL(
        plan->rdown == reader_down_weights.has_value(), "flat_combine_overlap: reader_down_weights iff reader tails");
    // the report args go past every writer's own (the factory checks): the gate/up cores' xy and the schedule's args
    cfg.cmb_rt = 48 + static_cast<uint32_t>(plan->gu.size()) + 2 * cfg.experts_per_chip;
    // the flat expert's rectangle, where combine multicasts `go`: it must stay off combine's rows (0-1 with untilizers,
    // the senders' row 0 with row-major y, which combine's readers read directly)
    if (std::getenv("FLAT_CMB_LOG")) {  // combine's cells on every chip (what MIMO_FL_XDOWN must avoid)
        namespace cmbp = ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;
        const auto placement =
            cmbp::decide_placement(device, combine_axis, combine_num_links, y_row_major ? 0 : cmbp::untilizers_per_group(), true);
        std::set<std::pair<uint32_t, uint32_t>> used;
        for (const auto& [coord, dp] : placement) {
            for (const auto& [stream, sp] : dp.streams) {
                used.insert({sp.worker_logical.x, sp.worker_logical.y});
            }
            for (const auto& g : dp.untilizers) {
                for (const auto& u : g) {
                    used.insert({u.logical.x, u.logical.y});
                }
            }
            used.insert({dp.collector->logical.x, dp.collector->logical.y});
        }
        std::string str;
        for (const auto& [x, y] : used) {
            str += fmt::format(" ({},{})", x, y);
        }
        log_info(tt::LogOp, "flat_combine_overlap: combine cells over all chips:{}", str);
    }
    // (MIMO_FL_XDOWN: extra down cores on combine's rows, which combine reaches by unicast instead)
    const auto cores = plan->arena_cores();
    const uint32_t first_row = y_row_major ? 1 : 2;
    CoreCoord lo{1000, 1000}, hi{0, 0};
    std::vector<CoreCoord> extra;
    for (const auto& c : cores) {
        if (c.y < first_row) {
            extra.push_back(c);
            continue;
        }
        lo = CoreCoord{std::min(lo.x, c.x), std::min(lo.y, c.y)};
        hi = CoreCoord{std::max(hi.x, c.x), std::max(hi.y, c.y)};
    }
    TT_FATAL(
        lo.y >= first_row,
        "flat_combine_overlap: the flat expert uses grid row {} (plan it with MIMO_FL_ROWS={},9)",
        lo.y,
        first_row);
    const uint32_t writers = static_cast<uint32_t>(plan->down.size() + (plan->rdown ? plan->rdn.size() : 0));
    const uint32_t rows = x.logical_shape()[-2];
    const ttnn::Tensor y = y_out ? *y_out
                                 : ttnn::empty(
                                       ttnn::Shape({rows, cfg.hidden}),
                                       y_row_major ? DataType::BFLOAT16 : DataType::BFLOAT8_B,
                                       y_row_major ? Layout::ROW_MAJOR : Layout::TILE,
                                       device,
                                       MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    const ttnn::Tensor out = ttnn::empty(
        ttnn::Shape({1, 1, seq_len_per_chip, num_experts_per_tok, cfg.hidden}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    // the arena over every worker core: the flat expert's buffers on its rows, combine's rings / control on rows 0-1
    const auto grid = device->compute_with_storage_grid_size();
    std::vector<CoreCoord> all;
    for (uint32_t yy = 0; yy < grid.y; ++yy) {
        for (uint32_t xx = 0; xx < grid.x; ++xx) {
            all.emplace_back(xx, yy);
        }
    }
    ttnn::Tensor arena = create_device_tensor(l1_sharded_spec(all, plan->arena_tiles * 32), device);
    ttnn::Tensor words = create_device_tensor(l1_sharded_spec(plan->relays, 32), device);
    FlatCombineOverlapParams p{
        .flat = cfg,
        .device = device,
        .experts_per_chip = cfg.experts_per_chip,
        .num_experts_per_tok = num_experts_per_tok,
        .seq_len_per_chip = seq_len_per_chip,
        .axis = combine_axis,
        .num_links = combine_num_links,
        .flat_cores = CoreRange(lo, hi),
        .extra_cores = extra,
        .writers = writers,
        .fwd_arrived = fwd_arrived_semaphore,
        .final_arrived = final_arrived_semaphore,
        .expert_go = expert_go_semaphore};
    const std::vector<const ttnn::Tensor*> bufs{
        &x,
        &counts,
        &regions,
        &global_expert_ids,
        &gate_up_weights,
        &down_weights,
        &done_words,
        &arena,
        &words,
        &y,
        &dispatched_metadata,
        &expert_offsets,
        &replicated_global_expert_idx_table,
        &out};
    for (const ttnn::Tensor* tt : bufs) {
        p.addrs.push_back(static_cast<uint32_t>(tt->buffer()->address()));
    }
    if (reader_down_weights) {
        p.addrs.push_back(static_cast<uint32_t>(reader_down_weights->buffer()->address()));
    }
    for (const auto* s : {&fwd_arrived_semaphore, &final_arrived_semaphore, &expert_go_semaphore}) {
        p.addrs.push_back(static_cast<uint32_t>(s->address()));
    }
    p.flat_key = {
        cfg.hidden,
        cfg.intermediate,
        cfg.experts_per_chip,
        cfg.num_global_experts,
        cfg.max_tokens,
        cfg.weights_bf8,
        cfg.activation,
        cfg.pin,
        cfg.down_fp32,
        cfg.x_bf16,
        cfg.h_bf16,
        cfg.y_row_major,
        cfg.cmb_rt,
        static_cast<uint32_t>(plan->down.size())};
    for (const auto& c : extra) {
        p.flat_key.push_back(static_cast<uint32_t>(c.x << 16 | c.y));
    }
    auto result = ttnn::prim::bringup::flat_combine_overlap(
        p,
        FlatCombineOverlapInputs{
            .x = x,
            .counts = counts,
            .regions = regions,
            .global_expert_ids = global_expert_ids,
            .gate_up_weights = gate_up_weights,
            .down_weights = down_weights,
            .reader_down_weights = reader_down_weights,
            .done_words = done_words,
            .arena = arena,
            .words = words,
            .y = y,
            .dispatched_metadata = dispatched_metadata,
            .expert_offsets = expert_offsets,
            .global_expert_idx_table = replicated_global_expert_idx_table,
            .output = out});
    arena.deallocate(true);
    words.deallocate(true);
    return result;
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
