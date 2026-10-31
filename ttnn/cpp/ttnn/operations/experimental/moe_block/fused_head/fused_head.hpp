// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.fused_head: the fused head: x all-gather, shared expert + reduce-scatter, router, routing
// exchange + planner, dispatch prep, latent (one program a chip). Reference: models/demos/moe_block/tt/fused_head.py
// HeadSE3 and moe_block.py FusedHead on mstaletovic/moe-trim.
//
// The host plan (device/head_plan.hpp) and the host model of the programs (device/head_program.hpp) are made once a
// block and kept in its resources; a call takes the resources' persistent words and tensors and launches the device
// operation (device/fused_head_device_operation.hpp).

#pragma once

#include <array>
#include <map>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/types.hpp"

#include "device/head_plan.hpp"
#include "device/head_program.hpp"
#include "ttnn/tensor/tensor.hpp"

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

// The head's outputs (the resources' persistent tensors, rewritten every call).
struct FusedHeadOutputs {
    Tensor
        se;  // the shared expert's reduced output: [rows / TP, H] (the owners' rows) or, with a latent, [rows, H / TP]
    Tensor rows;                   // the owners' joint routing rows [RB, 64 K] uint16
    Tensor xbuf;                   // the dispatch buffer [N rows / TP, H or latent] (this chip's owned slots written)
    Tensor route;                  // the owned tokens' route words
    std::optional<Tensor> latent;  // the owners' latent rows [rows / TP, latent] (a latent model)
    // the combine plan the routing planner writes (the routed experts' combine reads it)
    Tensor counts, regions, token_index, table, plan;
};

// The shared expert's and the router's weights in the head's layouts (prepare_fused_head_weights).
struct FusedHeadWeights {
    Tensor w;             // the engine's per-core W regions, bank-sharded (DRAM)
    Tensor w_router;      // W_r^T [E, H / TP] a chip, bf16 TILE
    Tensor router_bias;   // [E, 32] fp32 TILE
    Tensor router_index;  // [E, 32] uint16 TILE
};

// The weights in the head's layouts, from host tensors (float32 or bf16, row-major, any rank with these trailing
// dims): gate / up [Is, H] (the shared expert's), down [H, Is], latent_down [Hr, H] (a latent model), router_weight
// [E, H] (E the router's candidates), router_bias [E]. Every chip gets its TP slice (the plan's positions).
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
    std::optional<int64_t> l1_free = std::nullopt);

// The head's planned layout, programs' host model and persistent words on the block's resources (made once; a call
// only reads them).
struct FusedHeadSetup {
    std::shared_ptr<const HeadModel> model;
    HeadProgramConfig config;
    HeadWordAddresses words;
    uint64_t serial = 0;
};
const FusedHeadSetup& setup_fused_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt);

// ttnn.experimental.moe_block.fused_head: x [1, 1, rows, H / TP] a chip (bf16 TILE, DRAM) -> the head's outputs.
// x_full: the DRAM fallback's gathered x [1, 1, rows, H] (its layout only).
FusedHeadOutputs fused_head(
    const Tensor& x,
    const FusedHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    const std::optional<Tensor>& x_full = std::nullopt,
    std::optional<int64_t> l1_free = std::nullopt);

// The per-chip program descriptors fused_head would launch (no launch): tests compare them with the reference's.
// buffers keep the arena's view alive while the descriptors are read.
struct FusedHeadPrograms {
    std::vector<std::pair<ttnn::MeshCoordinate, tt::tt_metal::ProgramDescriptor>> programs;
    std::vector<tt::tt_metal::WorkloadBuffer> buffers;
    // every buffer the programs address: name -> (address, bytes) - the arena, the persistent words, the tensors
    std::map<std::string, std::pair<uint32_t, uint32_t>> addresses;
    // "r,c,kernel file,x,y" -> where the core's fabric connection args start (main's; the reference's differ)
    std::map<std::string, uint32_t> fabric_starts;
};
FusedHeadPrograms fused_head_programs(
    const Tensor& x,
    const FusedHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    const std::optional<Tensor>& x_full = std::nullopt,
    std::optional<int64_t> l1_free = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
