// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.fused_head: the fused head: x all-gather, shared expert + reduce-scatter, router, routing
// exchange + planner, dispatch prep (one program a chip).
//
// The host plan (device/head_plan.hpp) and the host model of the programs (device/head_program.hpp) are made once a
// block and kept in its resources; a call takes the resources' persistent words and tensors and launches the device
// operation (device/fused_head_device_operation.hpp).

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/types.hpp"

#include "device/head_plan.hpp"
#include "device/fused_head_device_operation_types.hpp"
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

// The free L1 a core the head plans on: l1_free, else the one measured at the block's first head (the L1 under the
// lowest occupied address then, less what the block keeps after planning, at most kL1Free - kL1Margin), kept in the
// resources so every later plan of the block uses the same figure.
int64_t fused_head_l1_free(MoEBlockResources& resources, std::optional<int64_t> l1_free = std::nullopt);

// The fused head's whole host plan on the block's resources, computed once and kept in the resources
// ("fused_head/plan/..."): model_checks, then head_plan_fit with config.head_dram_fallback, at
// fused_head_l1_free(resources, l1_free) - the plan the op runs. cb_floor: the other static CBs under the head's arena
// (default: the block's L1 plan's head_cb_floor when the resources hold one, else 0).
const HeadFit& plan_fused_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt,
    std::optional<int64_t> cb_floor = std::nullopt);

// The head's outputs (the resources' persistent tensors, rewritten every call).
struct FusedHeadOutputs {
    Tensor
        se;  // the shared expert's reduced output: [rows / TP, H] (the owners' rows) or, with a latent, [rows, H / TP]
    Tensor rows;                   // the owners' joint routing rows [RB, 64 K] uint16
    Tensor xbuf;                   // the dispatch buffer [N rows / TP, H or latent] (this chip's owned slots written)
    Tensor route;                  // the owned tokens' route words
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
// only reads them). l1_free: the free L1 a core the head plans on (default: measured on the mesh now, as
// FusedHead.env - the L1 left under the lowest occupied address, less what the block keeps after planning).
struct FusedHeadSetup {
    std::shared_ptr<const HeadModel> model;
    HeadProgramConfig config;
    std::vector<uint32_t> words;  // the persistent words' addresses, head_words' order
    FusedHeadInputs inputs;       // the persistent tensors (a call sets x and the weights)
    std::shared_ptr<HeadPatchStore> patches;
    std::shared_ptr<HeadPatchStore> patches_padded;  // the padded chunk's programs (actual_start / end): their args
    mutable std::optional<Tensor> arena;  // the default arena (made on first use)
};
const FusedHeadSetup& setup_fused_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt);

// ttnn.experimental.moe_block.fused_head: x [1, 1, rows, H / TP] a chip (bf16 TILE, DRAM) -> the head's outputs.
// x_full: the DRAM fallback's gathered x [1, 1, rows, H] (its layout only). arena: an L1 tensor over the head's arena
// cores (head_arena_cores, a shard of at least arena_bytes each) the head runs in - e.g. one the block shares between
// its ops; default one the resources keep. actual_start / actual_end: a padded chunk - main's chunked-prefill metadata
// scalars (the chunk's first real token's absolute position, one past its last; one uint32 each in DRAM, replicated),
// read on device each call (traceable: replays take the values of the moment). The chunk is in main's KV-pad-aware
// rotated layout over the SP rows (moe_padding_config): a row's real rows are a prefix, TP chip p owns its rows
// p S ..; padding tokens get no routes, so the dispatch, the routed experts and the combine never see them (their
// combined rows are zero). Absent: every row is real (the programs are the unpadded ones).
FusedHeadOutputs fused_head(
    const Tensor& x,
    const FusedHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    const std::optional<Tensor>& x_full = std::nullopt,
    std::optional<int64_t> l1_free = std::nullopt,
    const std::optional<Tensor>& arena = std::nullopt,
    const std::optional<Tensor>& actual_start = std::nullopt,
    const std::optional<Tensor>& actual_end = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
