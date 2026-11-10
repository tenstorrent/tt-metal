// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tracy/Tracy.hpp>
#include "fused_tail.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/experimental/moe_block/common/host_floats.hpp"
#include "ttnn/operations/experimental/moe_block/common/l1_budget.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "device/fused_tail_device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

namespace fused_tail_entry_detail {

using EnginePlanPtr = std::shared_ptr<const TailEnginePlan>;
using A2APlanPtr = std::shared_ptr<const TailA2APlan>;

uint32_t tile_bytes(DataType dtype) {
    TT_FATAL(
        dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT16,
        "fused_tail: W_up as bfloat8_b or bfloat16, got {}",
        dtype);
    return dtype == DataType::BFLOAT8_B ? 1088 : 2048;
}

// A mesh row's tokens: the chunk (one token loop's) / SP.
uint32_t rows_of(const MoEBlockResources& resources) {
    TT_FATAL(resources.config().chunk_tokens != 0, "fused_tail: MoEBlockConfig.chunk_tokens is not set");
    return resources.config().loop_tokens() / resources.topology().R;
}

const EnginePlanPtr& engine_plan(MoEBlockResources& resources, uint32_t wtb) {
    const auto& dims = resources.dims();
    const uint32_t G = resources.topology().C;
    TT_FATAL(dims.latent_dim.has_value(), "fused_tail: the engine is a latent model's tail");
    TT_FATAL(dims.latent_norm, "fused_tail: the engine always rms-normalizes the latent (latent_norm off)");
    TT_FATAL(dims.emb_dim % (32 * G) == 0, "fused_tail: H {} over TP {}: whole tiles", dims.emb_dim, G);
    const auto& plan = resources.plan<EnginePlanPtr>("fused_tail/engine_plan", [&] {
        return make_tail_engine_plan(
            resources,
            rows_of(resources),
            *dims.latent_dim,
            dims.emb_dim / G,
            wtb,
            free_cb_l1_bytes(resources.mesh_device(), kL1KeptLater, kArenaMax));
    });
    TT_FATAL(
        plan->geom.wtb == wtb, "fused_tail: the engine was planned for {} B W' tiles, got {}", plan->geom.wtb, wtb);
    return plan;
}

}  // namespace fused_tail_entry_detail
using namespace fused_tail_entry_detail;

Tensor fused_tail(
    const Tensor& y_own,
    const std::optional<Tensor>& shared_expert,
    MoEBlockResources& resources,
    const std::optional<Tensor>& w_up,
    tt::tt_metal::MathFidelity math_fidelity,
    const std::optional<Tensor>& output,
    bool all_loops,
    const std::optional<Tensor>& actual_start,
    const std::optional<Tensor>& actual_end,
    const std::optional<Tensor>& metadata,
    uint32_t loop) {
    ZoneScopedN("mb tail: entry");
    const auto& dims = resources.dims();
    const uint32_t loops = all_loops ? std::max<uint32_t>(1, resources.config().loops()) : 1;
    TT_FATAL(y_own.logical_shape()[-2] % loops == 0, "fused_tail: y_own's rows over {} token loops", loops);
    FusedTailParams params{
        .plan_key = resources.plan_key(),
        .math_fidelity = math_fidelity,
        .loops = loops,
        .output_memory_config = output.has_value() ? output->memory_config() : ttnn::DRAM_MEMORY_CONFIG,
    };
    FusedTailInputs inputs{.y_own = y_own, .shared_expert = shared_expert, .w_up = w_up, .output = output};
    // (a padded forward - fused_head's span: TailA2A skips its padding row tiles; the engine computes every row)
    TT_FATAL(
        !metadata.has_value() || (!actual_start.has_value() && !actual_end.has_value()),
        "fused_tail: metadata or actual_start / actual_end");
    TT_FATAL(actual_start.has_value() == actual_end.has_value(), "fused_tail: actual_start and actual_end together");
    const auto& cfg = resources.config();
    TT_FATAL(
        loop < cfg.loops() * cfg.splits() && (loops == 1 || loop % loops == 0),
        "fused_tail: loop {} of {} splits x {} token loops",
        loop,
        cfg.splits(),
        cfg.loops());
    if (!dims.latent_dim.has_value() && (actual_start.has_value() || metadata.has_value())) {
        TT_FATAL(
            std::max<uint32_t>(1, cfg.forward_chunks) == 1 || metadata.has_value(),
            "fused_tail: a forward of {} chunks takes metadata",
            cfg.forward_chunks);
        inputs.actual_start = actual_start;
        inputs.actual_end = actual_end;
        inputs.metadata = metadata;
        params.loop = loop;
        params.fwd_chunks = std::max<uint32_t>(1, cfg.forward_chunks);
        params.row_tokens = cfg.chunk_tokens / resources.topology().R;
    }
    if (dims.latent_dim.has_value()) {
        TT_FATAL(w_up.has_value(), "fused_tail: a latent model's tail needs w_up (prepare_fused_tail_weights)");
        TT_FATAL(math_fidelity != tt::tt_metal::MathFidelity::LoFi, "fused_tail: the up projection is never LoFi");
        const auto& plan = engine_plan(resources, tile_bytes(w_up->dtype()));
        const auto& g = plan->geom;
        params.kind = TailKind::Engine;
        params.nslot = g.NSLOT;
        params.engine_plan = plan;
        // (the ports' ready: the fence's one persistent word - the engine's per-step words are a per-call CB, zeroed
        // each call behind the engine's zero gate, so no L1_SMALL grows with the rows)
        inputs.ready = resources.persistent_words("fused_tail/engine_ready", plan->port_cores, 4);
    } else {
        TT_FATAL(!w_up.has_value(), "fused_tail: a model without a latent takes no w_up");
        const auto& plan = resources.plan<A2APlanPtr>("fused_tail/a2a_plan", [&] {
            return make_tail_a2a_plan(resources, rows_of(resources), dims.emb_dim);
        });
        const auto& g = plan->geom;
        params.kind = TailKind::A2A;
        params.a2a_plan = plan;
        // (the fence's arrival and ready words; the z-ready words are a per-call CB on the ports)
        inputs.arrival = resources.persistent_words("fused_tail/a2a_arrival", plan->port_union, 4);
        inputs.ready = resources.persistent_words("fused_tail/a2a_ready", plan->port_union, 4);
        inputs.staging = resources.persistent_tensor(
            "fused_tail/a2a_z",
            tt::tt_metal::TensorSpec(
                ttnn::Shape({1, 1, g.S, g.H}),
                tt::tt_metal::TensorLayout(
                    DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)));
    }
    return ttnn::prim::moe_block_fused_tail(params, std::move(inputs));
}

void plan_fused_tail(MoEBlockResources& resources, DataType dtype) {
    const auto& dims = resources.dims();
    const uint32_t rows = rows_of(resources);
    if (dims.latent_dim.has_value()) {
        (void)make_tail_engine_plan(
            resources,
            rows,
            *dims.latent_dim,
            dims.emb_dim / resources.topology().C,
            tile_bytes(dtype),
            free_cb_l1_bytes(resources.mesh_device(), kL1KeptLater, kArenaMax));
    } else {
        (void)make_tail_a2a_plan(resources, rows, dims.emb_dim);
    }
}

Tensor prepare_fused_tail_weights(
    const Tensor& latent_up, const Tensor& latent_norm, MoEBlockResources& resources, DataType dtype) {
    auto& mesh = resources.mesh_device();
    const auto& dims = resources.dims();
    const auto& topo = resources.topology();
    const auto& plan = *engine_plan(resources, tile_bytes(dtype));
    const auto& g = plan.geom;
    const uint32_t H = dims.emb_dim, Hr = g.Hr, Ho = g.Ho;
    TT_FATAL(
        latent_up.logical_volume() == static_cast<uint64_t>(H) * Hr && latent_up.logical_shape()[-1] == Hr,
        "fused_tail: latent_up must be [H {}, Hr {}], got {}",
        H,
        Hr,
        latent_up.logical_shape());
    TT_FATAL(latent_norm.logical_volume() == Hr, "fused_tail: latent_norm must be [Hr {}]", Hr);
    const auto up = to_floats(latent_up, "fused_tail");
    const auto gn = to_floats(latent_norm, "fused_tail");

    // Region (h, cx, k range p) = [k1 - k0][pn] tiles of half h's K rows of column cx's output columns, padded to
    // region_tiles; regions dealt over the banks (region i in bank i % banks, slot i / banks): one burst each.
    const uint32_t banks = plan.nbank, wreg = g.region_tiles();
    const uint32_t nreg = 2 * g.GX * g.PAIRS, per = (nreg + banks - 1) / banks;
    const uint32_t host_rows = per * wreg * 32, host_cols = banks * 32;
    const tt::tt_metal::MemoryConfig mc{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {banks - 1, 0})),
            {host_rows, 32},
            tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, host_rows, host_cols}),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), mc));
    std::vector<Tensor> shards;
    for (uint32_t r = 0; r < topo.R; ++r) {
        for (uint32_t c = 0; c < topo.C; ++c) {
            std::vector<float> host(static_cast<size_t>(host_rows) * host_cols, 0.0f);
            uint32_t i = 0;  // region index
            for (const auto& [h0, h1] : g.kh) {
                for (uint32_t cx = 0; cx < g.GX; ++cx) {
                    for (const auto& [k0, k1] : g.wk) {
                        const uint32_t bank = i % banks, slot = i / banks;
                        uint32_t t = 0;  // tile in the region
                        for (uint32_t kk = k0; kk < k1; ++kk) {
                            for (uint32_t j = 0; j < g.pn[cx]; ++j, ++t) {
                                for (uint32_t a = 0; a < 32; ++a) {
                                    const uint32_t k = (h0 + kk) * 32 + a;  // (latent row)
                                    const size_t row = static_cast<size_t>(slot * wreg + t) * 32 + a;
                                    for (uint32_t b = 0; b < 32; ++b) {
                                        const uint32_t n = c * Ho + (g.n0[cx] + j) * 32 + b;  // (output column)
                                        host[row * host_cols + bank * 32 + b] =
                                            up[static_cast<size_t>(n) * Hr + k] * gn[k];
                                    }
                                }
                            }
                        }
                        ++i;
                    }
                }
            }
            shards.push_back(Tensor::from_vector(std::move(host), spec));
        }
    }
    return ttnn::distributed::from_host_shards(to_mesh_order(std::move(shards), topo), mesh.shape()).to_device(&mesh, mc);
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
