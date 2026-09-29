// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert.hpp"

#include <map>
#include <mutex>
#include <tuple>

#include "device/flat_routed_expert_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& c) {
    using Key = std::tuple<const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t>;
    static std::mutex mu;
    static std::map<Key, std::shared_ptr<const FlatRoutedExpertPlan>> cache;
    const Key key{
        device, c.hidden, c.intermediate, c.experts_per_chip, c.num_global_experts, c.max_tokens, c.weights_bf8, c.pin};
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
    bool y_row_major) {
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
        .y_row_major = y_row_major};
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
    auto y = ttnn::prim::flat_routed_expert(
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
