// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_head.hpp"

#include <algorithm>
#include <atomic>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "device/fused_head_device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

std::string model_checks(const MoEModelDims& dims, const MoEDtypes& dtypes) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::MathFidelity;
    if (!dims.shared_expert_hidden_dim) {
        return "no shared expert (the fused head is built around the shared expert's matmul cores)";
    }
    const auto& rc = dims.router;
    if (rc.score_func != RouterScoreFunc::Sigmoid && rc.score_func != RouterScoreFunc::SqrtSoftplus) {
        return "router scores softmax: the fused router computes sigmoid / sqrtsoftplus";
    }
    if (!rc.normalize_topk) {
        return "router: unnormalized top-K weights (the fused router normalizes)";
    }
    if (rc.expert_groups != 1) {
        return "router: group-limited routing (" + std::to_string(rc.expert_groups) + " groups, top " +
               std::to_string(rc.topk_groups) + ") - the fused router's top-K is over every expert";
    }
    const auto w = dtypes.shared_weights;
    if (w != DataType::BFLOAT4_B && w != DataType::BFLOAT8_B && w != DataType::BFLOAT16) {
        return "shared expert weights: bf4 / bf8 / bf16";
    }
    return "";
}

HeadShape head_shape_of(
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MeshTopology& topology,
    std::array<uint32_t, 2> grid,
    uint32_t payload,
    uint32_t accumulators,
    tt::tt_metal::DataType shared_weights) {
    using tt::tt_metal::DataType;
    HeadShape s;
    s.rows = config.chunk_tokens / topology.R;
    s.H = dims.emb_dim;
    s.I = dims.shared_expert_hidden_dim.value_or(0) / topology.C;
    s.G = topology.C;
    s.E = dims.num_experts * dims.router.fold;
    s.K = dims.top_k;
    s.latent = dims.latent_dim;
    s.fold = dims.router.fold;
    s.links = config.num_links;
    s.ring = topology.wrap_x;
    s.grid = {std::min<uint32_t>(12, grid[0]), grid[1]};
    s.R = topology.R;
    s.sp_ring = topology.wrap_y;
    s.payload = payload;
    s.A = accumulators;
    s.w_dtype = shared_weights == DataType::BFLOAT4_B  ? WeightDtype::Bf4
                : shared_weights == DataType::BFLOAT16 ? WeightDtype::Bf16
                                                       : WeightDtype::Bf8;
    return s;
}

const HeadFit& plan_fused_head(
    MoEBlockResources& resources, const MoEDtypes& dtypes, uint32_t accumulators, std::optional<int64_t> l1_free) {
    const int64_t l1 = l1_free.value_or(kL1Free - kL1Margin);
    const std::string name = "fused_head/plan/A" + std::to_string(accumulators) + "/l1_" + std::to_string(l1) + "/w" +
                             std::to_string(static_cast<int>(dtypes.shared_weights));
    return resources.plan<HeadFit>(name, [&] {
        HeadFit f;
        f.why = model_checks(resources.dims(), dtypes);
        if (!f.why.empty()) {
            return f;
        }
        const auto g = resources.mesh_device().compute_with_storage_grid_size();
        const HeadShape shape = head_shape_of(
            resources.dims(),
            resources.config(),
            resources.topology(),
            {uint32_t(g.x), uint32_t(g.y)},
            uint32_t(resources.fabric_max_payload_bytes()),
            accumulators,
            dtypes.shared_weights);
        HeadFitOptions o;
        o.l1_free = l1;
        o.fallback = resources.config().head_dram_fallback;
        return head_plan_fit(shape, o);
    });
}

namespace fused_head_entry_detail {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorSpec;

TensorSpec dram_spec(uint32_t rows, uint32_t cols, DataType dt, Layout layout) {
    return TensorSpec(
        ttnn::Shape({1, 1, rows, cols}),
        tt::tt_metal::TensorLayout(dt, tt::tt_metal::PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG));
}

std::string key_of(uint32_t accumulators, int64_t l1, DataType w) {
    return "A" + std::to_string(accumulators) + "/l1_" + std::to_string(l1) + "/w" +
           std::to_string(static_cast<int>(w));
}

std::vector<ttnn::MeshCoordinate> all_coords(const tt::tt_metal::distributed::MeshDevice& mesh) {
    std::vector<ttnn::MeshCoordinate> out;
    for (uint32_t r = 0; r < mesh.shape()[0]; ++r) {
        for (uint32_t c = 0; c < mesh.shape()[1]; ++c) {
            out.emplace_back(r, c);
        }
    }
    return out;
}

FusedHeadInputs inputs_of(
    const Tensor& x,
    const FusedHeadWeights& w,
    MoEBlockResources& res,
    const HeadModel& model,
    const std::optional<Tensor>& x_full) {
    const auto s = head_tensor_shapes(model);
    auto t = [&](const std::string& name, const TensorSpec& spec) {
        return res.persistent_tensor("fused_head/" + name, spec);
    };
    FusedHeadInputs in{
        .x = x,
        .x_full = x_full,
        .w = w.w,
        .w_router = w.w_router,
        .router_bias = w.router_bias,
        .router_index = w.router_index,
        .y = t("y", dram_spec(s.y_rows, s.y_cols, DataType::BFLOAT16, Layout::TILE)),
        .rows = t("rows", dram_spec(s.rows_rows, s.rows_cols, DataType::UINT16, Layout::ROW_MAJOR)),
        .xbuf = t("xbuf", dram_spec(s.xbuf_rows, s.xbuf_cols, DataType::BFLOAT16, Layout::ROW_MAJOR)),
        .route = t("route", dram_spec(1, s.route_words, DataType::UINT32, Layout::ROW_MAJOR)),
        .latent = std::nullopt,
        .slots = t("slots", dram_spec(s.slots_rows, s.slot_words, DataType::UINT32, Layout::ROW_MAJOR)),
        .fslots = t("fslots", dram_spec(s.fslots_rows, s.slot_words, DataType::UINT32, Layout::ROW_MAJOR)),
        .counts =
            t("counts", dram_spec(s.plan_outs[0].first, s.plan_outs[0].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .regions =
            t("regions", dram_spec(s.plan_outs[1].first, s.plan_outs[1].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .token_index = t(
            "token_index", dram_spec(s.plan_outs[2].first, s.plan_outs[2].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .table =
            t("table", dram_spec(s.plan_outs[3].first, s.plan_outs[3].second, DataType::UINT32, Layout::ROW_MAJOR)),
        .plan = t("plan", dram_spec(s.plan_outs[4].first, s.plan_outs[4].second, DataType::UINT32, Layout::ROW_MAJOR)),
    };
    if (s.latent_rows) {
        in.latent = t("latent", dram_spec(s.latent_rows, s.latent_cols, DataType::BFLOAT16, Layout::TILE));
    }
    return in;
}

std::vector<float> to_floats(const Tensor& t) {
    TT_FATAL(t.storage_type() == StorageType::HOST, "fused_head: the weights come from the host");
    if (t.dtype() == DataType::FLOAT32) {
        return t.to_vector<float>();
    }
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "fused_head: host weights as float32 or bfloat16, got {}", t.dtype());
    const auto v = t.to_vector<bfloat16>();
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

// The free L1 a core the head plans on when none is given (FusedHead.env): the L1 from the allocator's base up to the
// lowest occupied address now, less what the block keeps after planning (the head's own words among it,
// FusedHead.BEFORE_HEAD), at most the planner's host figure, less its margin.
int64_t measured_free_l1(const tt::tt_metal::distributed::MeshDevice& mesh) {
    constexpr int64_t kBeforeHead = 56 * 1024;
    const int64_t base = int64_t(mesh.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const int64_t top = int64_t(mesh.lowest_occupied_compute_l1_address().value_or(base + mesh.l1_size_per_core()));
    return std::min(kL1Free, top - base - kBeforeHead) - kL1Margin;
}

}  // namespace fused_head_entry_detail

const FusedHeadSetup& setup_fused_head(
    MoEBlockResources& resources, const MoEDtypes& dtypes, uint32_t accumulators, std::optional<int64_t> l1_free) {
    using namespace fused_head_entry_detail;
    // (measured once, at the block's first head: later calls keep its plan whatever L1 the block took since)
    const int64_t l1 = l1_free.value_or(
        resources.plan<int64_t>("fused_head/l1_free", [&] { return measured_free_l1(resources.mesh_device()); }));
    const std::string key = key_of(accumulators, l1, dtypes.shared_weights);
    return resources.plan<FusedHeadSetup>("fused_head/setup/" + key, [&] {
        const HeadFit& fit = plan_fused_head(resources, dtypes, accumulators, l1);
        TT_FATAL(fit.ok, "fused_head: the head does not fit: {}", fit.why);
        const auto& dims = resources.dims();
        const auto& cfg = resources.config();
        auto& mesh = resources.mesh_device();
        const auto g = mesh.compute_with_storage_grid_size();
        FusedHeadSetup setup;
        HeadProgramConfig& pc = setup.config;
        pc.shape = head_shape_of(
            dims,
            cfg,
            resources.topology(),
            {uint32_t(g.x), uint32_t(g.y)},
            uint32_t(resources.fabric_max_payload_bytes()),
            accumulators,
            dtypes.shared_weights);
        pc.lay = fit.planned[0].lay;
        pc.glu = dims.shared_expert_activation.value_or(ExpertActivation{});
        pc.fidelity = dtypes.shared_fidelity;
        pc.w_dtype = dtypes.shared_weights;
        pc.route_scale = dims.router.route_scale;
        pc.sqrtsoftplus = dims.router.score_func == RouterScoreFunc::SqrtSoftplus;
        pc.ordered = cfg.deterministic;
        pc.tie_rule = cfg.tie_rule;
        pc.handoff_bytes = cfg.handoff_bytes();
        pc.l1_free = l1;
        setup.model = make_head_model(mesh, pc, resources.port_luts());
        for (const auto& spec : head_words(*setup.model)) {
            const auto& t = resources.persistent_words("fused_head/" + key + "/" + spec.name, spec.cores, spec.words);
            setup.words[spec.name] = static_cast<uint32_t>(t.buffer()->address());
        }
        static std::atomic<uint64_t> serials{0};
        setup.serial = ++serials;
        return setup;
    });
}

FusedHeadWeights prepare_fused_head_weights(
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
    using namespace fused_head_entry_detail;
    const auto& setup = setup_fused_head(resources, dtypes, accumulators, l1_free);
    auto w = prepare_head_weights(
        *setup.model,
        to_floats(gate),
        to_floats(up),
        to_floats(down),
        latent_down ? to_floats(*latent_down) : std::vector<float>{},
        to_floats(router_weight),
        to_floats(router_bias));
    return FusedHeadWeights{.w = w.w, .w_router = w.rw, .router_bias = w.rb, .router_index = w.rix};
}

FusedHeadOutputs fused_head(
    const Tensor& x,
    const FusedHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    const std::optional<Tensor>& x_full,
    std::optional<int64_t> l1_free) {
    using namespace fused_head_entry_detail;
    const auto& setup = setup_fused_head(resources, dtypes, accumulators, l1_free);
    FusedHeadParams params{
        .config = setup.config,
        .plan_key = resources.plan_key(),
        .resources_serial = setup.serial,
        .model = setup.model,
        .words = setup.words};
    auto in = inputs_of(x, weights, resources, *setup.model, x_full);
    FusedHeadOutputs out{
        .se = in.y,
        .rows = in.rows,
        .xbuf = in.xbuf,
        .route = in.route,
        .latent = in.latent,
        .counts = in.counts,
        .regions = in.regions,
        .token_index = in.token_index,
        .table = in.table,
        .plan = in.plan};
    ttnn::prim::moe_block_fused_head(params, std::move(in));
    return out;
}

FusedHeadPrograms fused_head_programs(
    const Tensor& x,
    const FusedHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    const std::optional<Tensor>& x_full,
    std::optional<int64_t> l1_free) {
    using namespace fused_head_entry_detail;
    const auto& setup = setup_fused_head(resources, dtypes, accumulators, l1_free);
    const auto in = inputs_of(x, weights, resources, *setup.model, x_full);
    FusedHeadPrograms out;
    auto w = build_head_workload(
        *setup.model, setup.words, head_buffers(in), all_coords(resources.mesh_device()), &out.fabric_starts);
    for (auto& [range, desc] : w.programs) {
        out.programs.emplace_back(range.start_coord(), std::move(desc));
    }
    out.buffers = std::move(w.buffers);
    const auto shapes = head_tensor_shapes(*setup.model);
    out.addresses["arena"] = {shapes.arena_base, shapes.arena_bytes};
    for (const auto& spec : head_words(*setup.model)) {
        out.addresses[spec.name] = {setup.words.at(spec.name), 4 * spec.words};
    }
    const std::vector<std::pair<std::string, const Tensor*>> ts = {
        {"x", &in.x},
        {"w", &in.w},
        {"rw", &in.w_router},
        {"rb", &in.router_bias},
        {"rix", &in.router_index},
        {"y", &in.y},
        {"rows", &in.rows},
        {"xbuf", &in.xbuf},
        {"route", &in.route},
        {"slots", &in.slots},
        {"fslots", &in.fslots},
        {"counts", &in.counts},
        {"regions", &in.regions},
        {"token_index", &in.token_index},
        {"table", &in.table},
        {"plan", &in.plan}};
    for (const auto& [n, t] : ts) {
        out.addresses[n] = {static_cast<uint32_t>(t->buffer()->address()), 1u << 30};
    }
    if (in.latent) {
        out.addresses["latent"] = {static_cast<uint32_t>(in.latent->buffer()->address()), 1u << 30};
    }
    if (in.x_full) {
        out.addresses["x_full"] = {static_cast<uint32_t>(in.x_full->buffer()->address()), 1u << 30};
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head
