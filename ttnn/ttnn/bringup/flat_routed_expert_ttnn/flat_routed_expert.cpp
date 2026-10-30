// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert.hpp"

#include <map>
#include <mutex>
#include <tuple>

#include "device/flat_routed_expert_device_operation.hpp"
#include "device/flat_with_program_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& c) {
    using Key = std::tuple<
        const void*,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        bool,
        uint32_t,
        bool,
        bool,
        std::vector<uint32_t>,
        uint32_t>;
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
        c.h_bf16,
        c.layout.key(),
        c.max_y_writers};
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        log_info(tt::LogOp, "flat_routed_expert plan: layout {}", c.layout.str());
        it = cache.emplace(key, std::make_shared<const FlatRoutedExpertPlan>(make_flat_routed_expert_plan(device, c)))
                 .first;
        log_info(
            tt::LogOp,
            "flat_routed_expert plan: gate/up row passes {} K group {} sub-block rows {}",
            it->second->gu_rp,
            it->second->gu_kg,
            it->second->mt * 32);
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
    bool h_bf16,
    bool y_bf16,
    uint32_t math_fidelity,
    const FlatLayoutOptions& layout,
    const std::optional<ttnn::Tensor>& output_tensor,
    bool weights_nk,
    uint32_t hidden,
    uint32_t max_y_writers) {
    using namespace tt::tt_metal;
    TT_FATAL(!(y_bf16 && y_row_major), "flat_routed_expert: y_bf16 is for tiled y (row-major y is bf16 already)");
    // hidden (0: x width x pages per row): x rows wider than H (x_pages_per_row 1), e.g. a dispatch buffer whose rows
    // carry extra columns after the token's H values; the relays read each row's first H values
    const uint32_t xw = x.logical_shape()[-1];
    TT_FATAL(
        hidden == 0 || (x_pages_per_row == 1 && hidden <= xw && hidden % 32 == 0 && !token_index),
        "flat_routed_expert: hidden {} needs x_pages_per_row 1, hidden <= x width {}, no token_index",
        hidden,
        xw);
    FlatRoutedExpertConfig cfg{
        .hidden = hidden ? hidden : xw * x_pages_per_row,
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
        .h_bf16 = h_bf16,
        .y_bf16 = y_bf16,
        .math_fidelity = math_fidelity,
        .layout = layout,
        .weights_nk = weights_nk,
        .max_y_writers = max_y_writers};
    auto* device = x.device();
    const auto plan = flat_routed_expert_plan(device, cfg);
    TT_FATAL(
        plan->rdown == reader_down_weights.has_value(),
        "flat_routed_expert: reader_down_weights iff the plan has reader tails");
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

ttnn::Tensor flat_routed_expert_with_program(
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
    const ttnn::Tensor& arena,
    const ttnn::Tensor& y,
    const tt::tt_metal::experimental::MeshProgramDescriptor& extra,
    uint64_t extra_key,
    const tt::tt_metal::CoreCoord& collector,
    uint32_t counts_addr,
    uint32_t writers,
    uint32_t activation,
    uint32_t pin,
    const std::optional<ttnn::Tensor>& token_index,
    bool down_fp32,
    bool x_bf16,
    bool h_bf16,
    const std::vector<std::vector<uint32_t>>& l1_args,
    const std::optional<ttnn::Tensor>& relay_words,
    std::optional<bool> y_l1,
    const FlatLayoutOptions& layout,
    uint32_t math_fidelity,
    bool weights_nk,
    bool patch_addresses,
    const std::vector<uint32_t>& extra_addrs,
    uint32_t max_y_writers) {
    using namespace tt::tt_metal;
    // (math_fidelity / weights_nk: as flat_routed_expert's, DeepSeek-V4.1 P.8; defaults 0 / false = the program before)
    FlatRoutedExpertConfig cfg{
        .hidden = x.logical_shape()[-1],
        .intermediate = intermediate,
        .experts_per_chip = static_cast<uint32_t>(global_expert_ids.logical_volume()),
        .num_global_experts = counts.logical_shape()[-1],
        .max_tokens = (max_tokens_per_expert + 31) / 32 * 32,
        .weights_bf8 = gate_up_weights.dtype() == DataType::BFLOAT8_B,
        .activation = activation,
        .pin = pin,
        .y_row_major = true,
        .down_fp32 = down_fp32,
        .pack_stochastic_rounding = false,
        .x_bf16 = x_bf16,
        .h_bf16 = h_bf16,
        .math_fidelity = math_fidelity,
        .layout = layout,
        .weights_nk = weights_nk,
        .max_y_writers = max_y_writers};
    auto* device = x.device();
    // (planned before cmb_rt is set: the plan cache ignores it, and the standalone plan keeps the DRISC senders)
    const auto plan = flat_routed_expert_plan(device, cfg);
    TT_FATAL(
        plan->rdown == reader_down_weights.has_value(),
        "flat_routed_expert_with_program: reader_down_weights iff the plan has reader tails");
    TT_FATAL(
        y.layout() == Layout::ROW_MAJOR && y.dtype() == DataType::BFLOAT16 &&
            y.logical_shape()[-2] >= (token_index ? token_index->logical_shape()[-1] : x.logical_shape()[-2]),
        "flat_routed_expert_with_program: y must be bf16 ROW_MAJOR with a row per dispatched row");
    const auto shard = arena.memory_config().shard_spec();
    TT_FATAL(
        shard.has_value() && shard->shape[0] >= plan->arena_tiles * 32,
        "flat_routed_expert_with_program: the arena needs >= {} rows per core",
        plan->arena_tiles * 32);
    for (const auto& c : plan->arena_cores()) {
        TT_FATAL(shard->grid.contains(c), "flat_routed_expert_with_program: the arena misses core {}", c.str());
    }
    cfg.cmb_rt = 48 + static_cast<uint32_t>(plan->gu.size()) + 2 * cfg.experts_per_chip;
    // the writers report local experts in slot order, with cumulative counts and no go handshake
    cfg.cmb_slot_order = true;
    cfg.cmb_no_go = true;
    // the y writers hand rows to the combine's L1 (SE_Y_L1; default: with hand-off args) - without args they write no y
    cfg.y_l1 = y_l1.value_or(!l1_args.empty());
    // (the relays' words: the caller's persistent tensor if given - a per-call one can land elsewhere, and its address
    // is part of the program-cache key a caller predicts; launches initialize it)
    ttnn::Tensor words = relay_words ? *relay_words : create_device_tensor(l1_sharded_spec(plan->relays, 32), device);
    if (relay_words) {  // (any L1 / L1_SMALL shards on the relays holding the x-ready word: e.g. 512 B a relay)
        const auto& mc = relay_words->memory_config();
        TT_FATAL(
            mc.is_sharded() &&
                (mc.buffer_type() == tt::tt_metal::BufferType::L1 ||
                 mc.buffer_type() == tt::tt_metal::BufferType::L1_SMALL) &&
                mc.shard_spec()->grid == rect_ranges(plan->relays) &&
                relay_words->buffer()->aligned_size_per_bank() >= X_READY_OFF + 4,
            "flat_routed_expert_with_program: relay_words must be L1 / L1_SMALL shards on the plan's relays, >= {} B "
            "each",
            X_READY_OFF + 4);
    }
    TT_FATAL(
        layout.x_ready <= 0 || 4 * plan->gu.size() <= X_READY_OFF,
        "flat_routed_expert_with_program: {} gate/up cores' freed words overrun the x-ready word at {}",
        plan->gu.size(),
        X_READY_OFF);
    FlatWithProgramParams p{
        .flat = cfg,
        // (an empty extra: the caller expects a program-cache hit, where the program is reused as is. A non-owning
        // view of the caller's descriptor, valid for this call: the op reads it only inside this launch - a copy is
        // a whole program set, ~7 ms of host a call)
        .extra = extra.mesh_programs.empty()
                     ? nullptr
                     : std::shared_ptr<const tt::tt_metal::experimental::MeshProgramDescriptor>(
                           std::shared_ptr<const tt::tt_metal::experimental::MeshProgramDescriptor>{}, &extra),
        .collector = collector,
        .counts_addr = counts_addr,
        .writers = writers,
        .extra_key = extra_key,
        .l1_args = l1_args,
        .patch_addresses = patch_addresses,
        .extra_addrs = extra_addrs};
    if (!patch_addresses) {  // (the address-keyed mode: every baked address in the key)
        for (const auto& v : l1_args) {
            p.addrs.insert(p.addrs.end(), v.begin(), v.end());
        }
        for (const ttnn::Tensor* tt : std::initializer_list<const ttnn::Tensor*>{
                 &x,
                 &counts,
                 &regions,
                 &global_expert_ids,
                 &gate_up_weights,
                 &down_weights,
                 &done_words,
                 &arena,
                 &words,
                 &y}) {
            p.addrs.push_back(static_cast<uint32_t>(tt->buffer()->address()));
        }
        for (const auto* o : {&reader_down_weights, &token_index}) {
            p.addrs.push_back(o->has_value() ? static_cast<uint32_t>((*o)->buffer()->address()) : 0u);
        }
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
        cfg.cmb_rt,
        static_cast<uint32_t>(arena.memory_config().shard_spec()->shape[0]),
        cfg.y_l1,
        cfg.cmb_no_go,
        cfg.cmb_slot_order};
    for (const uint32_t v : cfg.layout.key()) {
        p.flat_key.push_back(v);
    }
    p.flat_key.push_back(cfg.math_fidelity);
    p.flat_key.push_back(static_cast<uint32_t>(cfg.weights_nk));
    p.flat_key.push_back(cfg.max_y_writers);
    auto result = ttnn::prim::bringup::flat_with_program(
        p,
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
            .output = y,
            .token_index = token_index});
    if (!relay_words) {  // (the caller's persistent words stay)
        words.deallocate(true);
    }
    return result;
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
