// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_tail.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/api.hpp"
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

// A mesh row's tokens: the chunk (one token loop's) / SP when the config has a chunk, else from the owner rows.
uint32_t rows_of(const MoEBlockResources& resources, std::optional<uint32_t> own_rows) {
    const auto& topo = resources.topology();
    if (resources.config().chunk_tokens != 0) {
        return resources.config().loop_tokens() / topo.R;
    }
    TT_FATAL(own_rows.has_value(), "fused_tail: MoEBlockConfig.chunk_tokens is not set");
    return *own_rows * topo.C;
}

const EnginePlanPtr& engine_plan(MoEBlockResources& resources, uint32_t wtb, std::optional<uint32_t> own_rows) {
    const auto& dims = resources.dims();
    const uint32_t G = resources.topology().C;
    TT_FATAL(dims.latent_dim.has_value(), "fused_tail: the engine is a latent model's tail");
    TT_FATAL(dims.latent_norm, "fused_tail: the engine always rms-normalizes the latent (latent_norm off)");
    TT_FATAL(dims.emb_dim % (32 * G) == 0, "fused_tail: H {} over TP {}: whole tiles", dims.emb_dim, G);
    const auto& plan = resources.plan<EnginePlanPtr>("fused_tail/engine_plan", [&] {
        return make_tail_engine_plan(
            resources,
            rows_of(resources, own_rows),
            *dims.latent_dim,
            dims.emb_dim / G,
            wtb,
            free_cb_l1_bytes(resources.mesh_device(), kL1KeptLater, kArenaMax));
    });
    TT_FATAL(
        plan->geom.wtb == wtb, "fused_tail: the engine was planned for {} B W' tiles, got {}", plan->geom.wtb, wtb);
    return plan;
}

std::vector<float> to_floats(const Tensor& t) {
    TT_FATAL(t.storage_type() == StorageType::HOST, "fused_tail: the weights come from the host");
    if (t.dtype() == DataType::FLOAT32) {
        return t.to_vector<float>();
    }
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "fused_tail: host weights as float32 or bfloat16, got {}", t.dtype());
    const auto v = t.to_vector<bfloat16>();
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

}  // namespace fused_tail_entry_detail
using namespace fused_tail_entry_detail;

Tensor fused_tail(
    const Tensor& y_own,
    const std::optional<Tensor>& shared_expert,
    MoEBlockResources& resources,
    const std::optional<Tensor>& w_up,
    tt::tt_metal::MathFidelity math_fidelity,
    const std::optional<Tensor>& output) {
    const auto& dims = resources.dims();
    const uint32_t own_rows = y_own.logical_shape()[-2];
    FusedTailParams params{
        .plan_key = resources.plan_key(),
        .math_fidelity = math_fidelity,
        .output_memory_config = output.has_value() ? output->memory_config() : ttnn::DRAM_MEMORY_CONFIG,
    };
    FusedTailInputs inputs{.y_own = y_own, .shared_expert = shared_expert, .w_up = w_up, .output = output};
    if (dims.latent_dim.has_value()) {
        TT_FATAL(w_up.has_value(), "fused_tail: a latent model's tail needs w_up (prepare_fused_tail_weights)");
        TT_FATAL(math_fidelity != tt::tt_metal::MathFidelity::LoFi, "fused_tail: the up projection is never LoFi");
        const auto& plan = engine_plan(resources, tile_bytes(w_up->dtype()), own_rows);
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
            return make_tail_a2a_plan(resources, rows_of(resources, own_rows), dims.emb_dim);
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
    const uint32_t rows = rows_of(resources, std::nullopt);
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
    const auto& plan = *engine_plan(resources, tile_bytes(dtype), std::nullopt);
    const auto& g = plan.geom;
    const uint32_t H = dims.emb_dim, Hr = g.Hr, Ho = g.Ho;
    TT_FATAL(
        latent_up.logical_volume() == static_cast<uint64_t>(H) * Hr && latent_up.logical_shape()[-1] == Hr,
        "fused_tail: latent_up must be [H {}, Hr {}], got {}",
        H,
        Hr,
        latent_up.logical_shape());
    TT_FATAL(latent_norm.logical_volume() == Hr, "fused_tail: latent_norm must be [Hr {}]", Hr);
    const auto up = to_floats(latent_up);
    const auto gn = to_floats(latent_norm);

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
    return ttnn::distributed::from_host_shards(shards, mesh.shape()).to_device(&mesh, mc);
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
