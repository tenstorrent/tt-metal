// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.fused_head: the fused head: x all-gather, shared expert + reduce-scatter, router, routing
// exchange + planner, dispatch prep, latent (one program a chip). Reference: models/demos/moe_block/tt/fused_head.py
// HeadSE3 and moe_block.py FusedHead on mstaletovic/moe-trim.
//
// Today: the head's host plan (device/head_plan.hpp, cached in the block's resources) and the kernels
// (device/kernels/); the device operation is not written yet (README.md).

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/types.hpp"

#include "device/head_plan.hpp"

namespace ttnn::operations::experimental::moe_block {
struct MoEModelDims;
struct MoEDtypes;
struct MoEBlockConfig;
struct MeshTopology;
class MoEBlockResources;
}  // namespace ttnn::operations::experimental::moe_block

namespace ttnn::operations::experimental::moe_block::fused_head {

// The model-level requirements of the fused head (FusedHead.model_checks): why not, or "".
std::string model_checks(const MoEModelDims& dims, const MoEDtypes& dtypes);

// The head_plan HeadShape of a block's head (FusedHead.shape): grid = the compute grid (x at most 12), payload = the
// fabric's packet payload bytes, accumulators = the combine's.
HeadShape head_shape_of(
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MeshTopology& topology,
    std::array<uint32_t, 2> grid,
    uint32_t payload,
    uint32_t accumulators,
    tt::tt_metal::DataType shared_weights = tt::tt_metal::DataType::BFLOAT8_B);

// The fused head's whole host plan on the block's resources (FusedHead.plan), computed once and kept in the resources
// ("fused_head/plan/..."): model_checks, then head_plan_fit with config.head_dram_fallback. l1_free: the free L1 a
// core the head may take (default the planner's host figure, kL1Free - kL1Margin).
const HeadFit& plan_fused_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
