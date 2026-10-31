// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_head.hpp"

#include <algorithm>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

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

}  // namespace ttnn::operations::experimental::moe_block::fused_head
