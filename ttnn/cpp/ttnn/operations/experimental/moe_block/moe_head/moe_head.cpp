// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_head.hpp"
#include <tracy/Tracy.hpp>

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/moe_block/common/host_floats.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/l1_budget.hpp"
#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "device/moe_head_device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

std::string model_checks(const MoEModelDims& dims, const MoEDtypes& dtypes) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::MathFidelity;
    if (!dims.shared_expert_hidden_dim) {
        return "no shared expert (the MoE head is built around the shared expert's matmul cores)";
    }
    const auto weights_dtype = dtypes.shared_weights;
    if (weights_dtype != DataType::BFLOAT4_B && weights_dtype != DataType::BFLOAT8_B &&
        weights_dtype != DataType::BFLOAT16) {
        return "shared expert weights: bf4 / bf8 / bf16";
    }
    return "";
}

HeadShape head_shape_of(
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MeshTopology& topology,
    const FirstHops& first_hops,
    std::array<uint32_t, 2> grid,
    uint32_t payload,
    uint32_t accumulators,
    tt::tt_metal::DataType shared_weights,
    uint32_t row_blocks) {
    using tt::tt_metal::DataType;
    HeadShape shape;
    shape.rows = config.split_tokens() / topology.R / row_blocks;
    shape.row_blocks = row_blocks;
    shape.splits = config.splits();
    shape.row_tokens = config.chunk_tokens / topology.R;  // (the caller's SP row: its real rows' closed form)
    shape.forward_chunks = std::max<uint32_t>(1, config.forward_chunks);
    shape.hidden_dim = dims.emb_dim;
    shape.shared_expert_dim = dims.shared_expert_hidden_dim.value_or(0) / topology.C;
    shape.tp_size = topology.C;
    shape.router_candidates = dims.num_experts * dims.router.fold;
    shape.top_k = dims.top_k;
    shape.latent = dims.latent_dim;
    shape.fold = dims.router.fold;
    shape.links = config.num_links;
    shape.ring = topology.wrap_x;
    shape.grid = grid;
    shape.sp_rows = topology.R;
    shape.sp_ring = topology.wrap_y;
    shape.transposed = topology.transposed;
    TT_FATAL(
        first_hops.N == topology.num_chips(),
        "moe_head: first hops for {} chips, the mesh has {}",
        first_hops.N,
        topology.num_chips());
    shape.first_hops = first_hops.dirs;
    shape.payload = payload;
    shape.accumulators = accumulators;
    shape.w_dtype = shared_weights == DataType::BFLOAT4_B  ? WeightDtype::Bf4
                    : shared_weights == DataType::BFLOAT16 ? WeightDtype::Bf16
                                                           : WeightDtype::Bf8;
    return shape;
}

namespace moe_head_l1_detail {

// The free L1 a core the head plans on when none is given: the block's rule (common/l1_budget.hpp
// free_l1_span_bytes: the L1 from the allocator's base up to the lowest occupied address now, with nothing allocated
// the allocator's L1 bank), less the planner's margin (what the block allocates in L1 after planning: 128 B a bank with
// its persistent words in L1_SMALL).
int64_t measured_free_l1(const tt::tt_metal::distributed::MeshDevice& mesh) {
    return int64_t(free_l1_span_bytes(mesh)) - kL1Margin;
}

}  // namespace moe_head_l1_detail

int64_t moe_head_l1_free(MoEBlockResources& resources, std::optional<int64_t> l1_free) {
    if (l1_free) {
        return *l1_free;
    }
    // (measured once, at the block's first head: later calls keep its plan whatever L1 the block took since)
    return resources.plan<int64_t>(
        "moe_head/l1_free", [&] { return moe_head_l1_detail::measured_free_l1(resources.mesh_device()); });
}

namespace moe_head_l1_detail {
// The other static CBs under the head's arena: the caller's, else the block's L1 plan's.
int64_t cb_floor_of(const MoEBlockResources& resources, std::optional<int64_t> cb_floor) {
    if (cb_floor) {
        return *cb_floor;
    }
    const auto l1_plan = block_l1_plan(resources);
    return l1_plan ? int64_t(l1_plan->head_cb_floor) : 0;
}
}  // namespace moe_head_l1_detail

namespace moe_head_l1_detail {
// The row-block counts a call's rows allow: whole tile rows a block, and (more than one block) whole pages of every
// DRAM bank a block for x and the shared expert's output, which the kernels step a block's stride a bank. The test-only
// MoEBlockConfig.force_head_row_blocks: that count alone.
std::vector<uint32_t> row_block_counts(MoEBlockResources& resources) {
    const auto& config = resources.config();
    if (config.force_head_row_blocks) {
        return {config.force_head_row_blocks};
    }
    const auto& topology = resources.topology();
    const uint32_t rows = config.split_tokens() / topology.R, tp_size = topology.C, hidden = resources.dims().emb_dim;
    const uint32_t banks = resources.mesh_device().allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const bool latent = resources.dims().latent_dim.has_value();
    constexpr uint32_t tile = tt::constants::TILE_HEIGHT;
    std::vector<uint32_t> out;
    for (uint32_t k = 1; k <= 64; ++k) {
        if (rows % (k * tile) != 0) {
            continue;
        }
        const uint32_t block_rows = rows / k;
        if (k > 1) {
            if (!latent && block_rows % (tp_size * tile) != 0) {
                continue;
            }
            const uint64_t x_pages = uint64_t(block_rows / tile) * (hidden / tp_size / tile);
            const uint64_t y_pages = latent ? x_pages : uint64_t(block_rows / tp_size / tile) * (hidden / tile);
            if (x_pages % banks != 0 || y_pages % banks != 0) {
                continue;
            }
        }
        out.push_back(k);
    }
    return out;
}
}  // namespace moe_head_l1_detail

const HeadFit& plan_moe_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free,
    std::optional<int64_t> cb_floor) {
    const int64_t free_l1 = moe_head_l1_free(resources, l1_free);
    const int64_t floor = moe_head_l1_detail::cb_floor_of(resources, cb_floor);
    const std::string name = "moe_head/plan/A" + std::to_string(accumulators) + "/l1_" + std::to_string(free_l1) +
                             "/floor_" + std::to_string(floor) + "/w" +
                             std::to_string(static_cast<int>(dtypes.shared_weights));
    const HeadFit& planned = resources.plan<HeadFit>(name, [&] {
        HeadFit refusal;
        refusal.reason = model_checks(resources.dims(), dtypes);
        if (!refusal.reason.empty()) {
            return refusal;
        }
        const auto grid = resources.mesh_device().compute_with_storage_grid_size();
        HeadFitOptions options;
        options.l1_free = free_l1;
        options.cb_floor = floor;
        options.min_router_iterations = std::max<uint32_t>(1, resources.config().head_router_iterations);
        // The row blocks: the call's rows at once wherever they fit, else the count whose plan fits with the least
        // predicted time (blocks x a block's). The kernels block over the blocks in one program.
        std::optional<HeadFit> first, best;
        double best_us = 0;
        for (const uint32_t k : moe_head_l1_detail::row_block_counts(resources)) {
            const HeadShape shape = head_shape_of(
                resources.dims(),
                resources.config(),
                resources.topology(),
                resources.first_hops(),
                {uint32_t(grid.x), uint32_t(grid.y)},
                uint32_t(resources.fabric_max_payload_bytes()),
                accumulators,
                dtypes.shared_weights,
                k);
            HeadFit fit = head_plan_fit(shape, options);
            fit.row_blocks = k;
            if (!first) {
                first = fit;
            }
            if (!fit.ok || fit.planned.empty()) {
                continue;
            }
            const double total_us = k * fit.planned[0].predicted_us;
            if (!best || total_us < best_us) {
                best = fit;
                best_us = total_us;
            }
            if (k == 1) {
                break;  // (one block wherever the rows fit at once)
            }
        }
        if (best) {
            return *best;
        }
        if (!first) {
            first = HeadFit{};
            first->reason = "no row-block count divides the call's rows into whole tiles a block";
        } else {
            first->reason += " (and no row-block count of the call's rows fits either)";
        }
        return *first;
    });
    return planned;
}

namespace moe_head_entry_detail {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorSpec;

TensorSpec dram_spec(uint32_t rows, uint32_t columns, DataType dtype, Layout layout) {
    return TensorSpec(
        ttnn::Shape({1, 1, rows, columns}),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG));
}

std::string key_of(uint32_t accumulators, int64_t free_l1, int64_t floor, DataType shared_weights) {
    return "A" + std::to_string(accumulators) + "/l1_" + std::to_string(free_l1) + "/floor_" + std::to_string(floor) +
           "/w" + std::to_string(static_cast<int>(shared_weights));
}

// The head's persistent tensors (its outputs and the routing exchange's), in an inputs template: the shared expert's
// output holds every row block's rows, block after block.
MoEHeadInputs persistent_inputs(MoEBlockResources& resources, const HeadModel& model, uint32_t row_blocks) {
    const auto shapes = head_tensor_shapes(model);
    auto tensor = [&](const std::string& name, const TensorSpec& spec) {
        return resources.persistent_tensor("moe_head/" + name, spec);
    };
    MoEHeadInputs inputs{
        .y = tensor("y", dram_spec(shapes.y_rows * row_blocks, shapes.y_columns, DataType::BFLOAT16, Layout::TILE)),
        .rows = tensor(
            "rows",
            dram_spec(shapes.routing_rows_rows, shapes.routing_rows_columns, DataType::UINT16, Layout::ROW_MAJOR)),
        .dispatch_buffer = tensor(
            "xbuf",
            dram_spec(
                shapes.dispatch_buffer_rows, shapes.dispatch_buffer_columns, DataType::BFLOAT16, Layout::ROW_MAJOR)),
        .route = tensor("route", dram_spec(1, shapes.route_words, DataType::UINT32, Layout::ROW_MAJOR)),
        .slots = tensor("slots", dram_spec(shapes.slots_rows, shapes.slot_words, DataType::UINT32, Layout::ROW_MAJOR)),
        .counts = tensor(
            "counts",
            dram_spec(
                shapes.plan_outputs[0].first, shapes.plan_outputs[0].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .regions = tensor(
            "regions",
            dram_spec(
                shapes.plan_outputs[1].first, shapes.plan_outputs[1].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .token_index = tensor(
            "token_index",
            dram_spec(
                shapes.plan_outputs[2].first, shapes.plan_outputs[2].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .table = tensor(
            "table",
            dram_spec(
                shapes.plan_outputs[3].first, shapes.plan_outputs[3].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .plan = tensor(
            "plan",
            dram_spec(
                shapes.plan_outputs[4].first, shapes.plan_outputs[4].second, DataType::UINT32, Layout::ROW_MAJOR)),
    };
    return inputs;
}

// A call's inputs: the template's persistent tensors with this call's x and weights.
MoEHeadInputs call_inputs(const MoEHeadSetup& setup, const Tensor& x, const MoEHeadWeights& weights) {
    ZoneScopedN("moe_head: call inputs");
    MoEHeadInputs inputs = setup.inputs;
    inputs.x = x;
    inputs.w = weights.w;
    inputs.w_router = weights.w_router;
    inputs.router_bias = weights.router_bias;
    inputs.router_index = weights.router_index;
    return inputs;
}

// The arena when the caller gives none: an L1 tensor the resources keep (the arena cores, a shard of arena_bytes
// each) - it holds the L1 for good; a block whose other ops need that L1 passes one arena it shares between them.
Tensor default_arena(MoEBlockResources& resources, const MoEHeadSetup& setup) {
    const auto& arena = setup.arena;
    if (arena.has_value()) {
        return *arena;
    }
    const auto cores = head_arena_cores(*setup.model);
    const uint32_t words = head_tensor_shapes(*setup.model).arena_bytes / 4;
    const MemoryConfig memory_config{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(cores, {1, words}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const TensorSpec spec(
        ttnn::Shape({cores.num_cores(), words}),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
    setup.arena = resources.persistent_tensor("moe_head/arena", spec);
    return *setup.arena;
}

}  // namespace moe_head_entry_detail

const MoEHeadSetup& setup_moe_head(
    MoEBlockResources& resources, const MoEDtypes& dtypes, uint32_t accumulators, std::optional<int64_t> l1_free) {
    ZoneScopedN("moe_head: setup lookup");
    using namespace moe_head_entry_detail;
    const int64_t free_l1 = moe_head_l1_free(resources, l1_free);
    const int64_t floor = moe_head_l1_detail::cb_floor_of(resources, std::nullopt);
    const std::string key = key_of(accumulators, free_l1, floor, dtypes.shared_weights);
    return resources.plan<MoEHeadSetup>("moe_head/setup/" + key, [&] {
        const HeadFit& fit = plan_moe_head(resources, dtypes, accumulators, free_l1, floor);
        TT_FATAL(fit.ok, "moe_head: the head does not fit: {}", fit.reason);
        const auto& dims = resources.dims();
        const auto& block_config = resources.config();
        auto& mesh = resources.mesh_device();
        const auto grid = mesh.compute_with_storage_grid_size();
        MoEHeadSetup setup;
        HeadProgramConfig& program_config = setup.config;
        program_config.shape = head_shape_of(
            dims,
            block_config,
            resources.topology(),
            resources.first_hops(),
            {uint32_t(grid.x), uint32_t(grid.y)},
            uint32_t(resources.fabric_max_payload_bytes()),
            accumulators,
            dtypes.shared_weights,
            fit.row_blocks);
        program_config.layout = fit.planned[0].layout;
        program_config.activation = dims.shared_expert_activation.value_or(ExpertActivation{});
        program_config.fidelity = dtypes.shared_fidelity;
        program_config.w_dtype = dtypes.shared_weights;
        program_config.route_scale = dims.router.route_scale;
        program_config.sqrt_softplus = dims.router.score_func == RouterScoreFunc::SqrtSoftplus;
        program_config.handoff_bytes = block_config.handoff_bytes();
        program_config.l1_free = free_l1;
        program_config.cb_floor = floor;
        setup.model = make_head_model(mesh, program_config, resources.port_luts());
        for (const auto& spec : head_words(*setup.model)) {
            const auto& words_tensor =
                resources.persistent_words("moe_head/" + key + "/" + spec.name, spec.cores, spec.words);
            setup.words.push_back(static_cast<uint32_t>(words_tensor.buffer()->address()));
        }
        setup.row_blocks = fit.row_blocks;
        setup.inputs = persistent_inputs(resources, *setup.model, fit.row_blocks);
        for (uint32_t i = 0; i < block_config.splits(); ++i) {
            setup.patches.push_back(std::make_shared<HeadPatchStore>());
            setup.patches_padded.push_back(std::make_shared<HeadPatchStore>());
        }
        return setup;
    });
}

MoEHeadWeights prepare_moe_head_weights(
    const Tensor& gate,
    const Tensor& up,
    const Tensor& down,
    const std::optional<Tensor>& latent_down,
    const Tensor& router_weight,
    const Tensor& router_bias,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free) {
    using namespace moe_head_entry_detail;
    const auto& setup = setup_moe_head(resources, dtypes, accumulators, l1_free);
    auto device_weights = prepare_head_weights(
        *setup.model,
        to_floats(gate, "moe_head"),
        to_floats(up, "moe_head"),
        to_floats(down, "moe_head"),
        latent_down ? to_floats(*latent_down, "moe_head") : std::vector<float>{},
        to_floats(router_weight, "moe_head"),
        to_floats(router_bias, "moe_head"));
    return MoEHeadWeights{
        .w = device_weights.w,
        .w_router = device_weights.router_w,
        .router_bias = device_weights.router_bias,
        .router_index = device_weights.router_index};
}

MoEHeadOutputs moe_head(
    const Tensor& x,
    const MoEHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free,
    const std::optional<Tensor>& arena,
    const std::optional<Tensor>& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t split,
    const std::optional<Tensor>& metadata) {
    ZoneScopedN("moe_head: entry");
    using namespace moe_head_entry_detail;
    TT_FATAL(
        resources.config().pad_tokens == 0 || actual_start.has_value() || metadata.has_value(),
        "moe_head: a chunk with padding rows (MoEBlockConfig.pad_tokens) takes actual_start / actual_end");
    TT_FATAL(split < resources.config().splits(), "moe_head: split {} of {}", split, resources.config().splits());
    const auto& setup = setup_moe_head(resources, dtypes, accumulators, l1_free);
    const bool padded = actual_start.has_value() || metadata.has_value();
    MoEHeadParams params{
        .config = setup.config,
        .plan_key = resources.plan_key(),
        .model = setup.model,
        .words = setup.words,
        .arena = arena ? *arena : default_arena(resources, setup),
        .patches = padded ? setup.patches_padded[split] : setup.patches[split],
        .row_blocks = setup.row_blocks,
        .split = split};
    auto inputs = call_inputs(setup, x, weights);
    inputs.actual_start = actual_start;
    inputs.metadata = metadata;
    inputs.actual_end = actual_end;
    MoEHeadOutputs out{
        .shared_expert_output = inputs.y,
        .routing_rows = inputs.rows,
        .dispatch_buffer = inputs.dispatch_buffer,
        .route_words = inputs.route,
        .counts = inputs.counts,
        .regions = inputs.regions,
        .token_index = inputs.token_index,
        .table = inputs.table,
        .plan = inputs.plan,
        .row_blocks = setup.row_blocks};
    ttnn::prim::moe_block_head(params, std::move(inputs));
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
