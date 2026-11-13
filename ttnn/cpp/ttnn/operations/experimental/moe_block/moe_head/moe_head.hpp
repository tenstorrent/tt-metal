// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.moe_head: the MoE head: x all-gather, shared expert + reduce-scatter, router, routing
// exchange + planner, dispatch prep (one program a chip).
//
// The host plan (device/head_plan.hpp) and the host model of the programs (device/head_program.hpp) are made once a
// block and kept in its resources; a call takes the resources' persistent words and tensors and launches the device
// operation (device/moe_head_device_operation.hpp).

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "ttnn/tensor/types.hpp"

#include "device/head_plan.hpp"
#include "device/moe_head_device_operation_types.hpp"
#include "device/head_program.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {
struct MoEModelDims;
struct MoEDtypes;
struct MoEBlockConfig;
struct MeshTopology;
class MoEBlockResources;
}  // namespace ttnn::operations::experimental::moe_block

namespace ttnn::operations::experimental::moe_block::moe_head {

// The model-level requirements of the MoE head: why not, or "".
std::string model_checks(const MoEModelDims& dims, const MoEDtypes& dtypes);

// The HeadShape (device/head_plan.hpp) of a block's head: grid = the compute grid (x at most 12), payload = the
// fabric's packet payload bytes, accumulators = the combine's, row_blocks = the rows of a call (one split's) as that
// many blocks of rows (the head's kernels block over them; plan_moe_head picks it).
HeadShape head_shape_of(
    const MoEModelDims& dims,
    const MoEBlockConfig& config,
    const MeshTopology& topology,
    const FirstHops& first_hops,
    std::array<uint32_t, 2> grid,
    uint32_t payload,
    uint32_t accumulators,
    tt::tt_metal::DataType shared_weights = tt::tt_metal::DataType::BFLOAT8_B,
    uint32_t row_blocks = 1);

// The free L1 a core the head plans on: l1_free, else the one measured at the block's first head (the L1 under the
// lowest occupied address then, less kL1Margin), kept in the resources so every later plan of the block uses the same
// figure.
int64_t moe_head_l1_free(MoEBlockResources& resources, std::optional<int64_t> l1_free = std::nullopt);

// The MoE head's whole host plan on the block's resources, computed once and kept in the resources
// ("moe_head/plan/..."): model_checks, then head_plan_fit for each row-block count the rows allow (HeadFit.row_blocks:
// 1 wherever the rows fit at once, else the count with the least predicted time), at
// moe_head_l1_free(resources, l1_free) - the plan the op runs. cb_floor: the other static CBs under the head's arena
// (default: the block's L1 plan's head_cb_floor when the resources hold one, else 0).
const HeadFit& plan_moe_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt,
    std::optional<int64_t> cb_floor = std::nullopt);

// The head's outputs (the resources' persistent tensors, rewritten every call).
struct MoEHeadOutputs {
    Tensor shared_expert_output;  // the shared expert's reduced output: [rows / TP, H] (the owners' rows) or, with a
                                  // latent, [rows, H / TP]
    Tensor routing_rows;          // the owners' joint routing rows [RB, 64 K] uint16
    Tensor dispatch_buffer;       // the dispatch buffer [N rows / TP, H or latent] (this chip's owned slots written)
    Tensor route_words;           // the owned tokens' route words
    // the combine plan the routing planner writes (the routed experts' combine reads it)
    Tensor counts, regions, token_index, table, plan;
    // the row blocks the head ran the call's rows as (its planner's pick: plan_moe_head's HeadFit.row_blocks). The
    // owner-order outputs (shared_expert_output without a latent, routing_rows, the dispatch's and the combine's rows)
    // are in its layout - block b's owned tokens after block b - 1's - so the tail takes it (moe_tail row_blocks).
    uint32_t row_blocks = 1;
};

// The shared expert's and the router's weights in the head's layouts (prepare_moe_head_weights).
struct MoEHeadWeights {
    Tensor w;             // the engine's per-core W regions, bank-sharded (DRAM)
    Tensor w_router;      // W_r^T [E, H / TP] a chip, bf16 TILE
    Tensor router_bias;   // [E, 32] fp32 TILE
    Tensor router_index;  // [E, 32] uint16 TILE
};

// The weights in the head's layouts, from host tensors (float32 or bf16, row-major, any rank with these trailing
// dims): gate / up [Is, H] (the shared expert's), down [H, Is], latent_down [Hr, H] (a latent model), router_weight
// [E, H] (E the router's candidates), router_bias [E]. Every chip gets its TP slice (the plan's positions).
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
    std::optional<int64_t> l1_free = std::nullopt);

// The head's planned layout, programs' host model and persistent words on the block's resources (made once; a call
// only reads them). l1_free: the free L1 a core the head plans on (default: measured on the mesh now, as
// moe_head_l1_free - the L1 left under the lowest occupied address, less what the block keeps after planning).
struct MoEHeadSetup {
    std::shared_ptr<const HeadModel> model;
    HeadProgramConfig config;
    std::vector<uint32_t> words;  // the persistent words' addresses, head_words' order
    MoEHeadInputs inputs;         // the persistent tensors (a call sets x and the weights)
    uint32_t row_blocks = 1;      // the planner's row blocks (HeadFit.row_blocks): inputs.y holds every block's rows
    // a split's program (MoEBlockConfig.chunk_splits: one each - a padded chunk's real rows depend on the split),
    // unpadded / padded (actual_start / end)
    std::vector<std::shared_ptr<HeadPatchStore>> patches;
    std::vector<std::shared_ptr<HeadPatchStore>> patches_padded;
    mutable std::optional<Tensor> arena;  // the default arena (made on first use)
};
const MoEHeadSetup& setup_moe_head(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt);

// ttnn.experimental.moe_block.moe_head: x [1, 1, rows, H / TP] a chip (bf16 TILE, DRAM) -> the head's outputs. arena:
// an L1 tensor over the head's arena cores (head_arena_cores, a shard of at least arena_bytes each) the head runs in -
// e.g. one the block shares between its ops; default one the resources keep. actual_start / actual_end: a padded chunk
// - main's chunked-prefill metadata scalars (the chunk's first real token's absolute position, one past its last; one
// uint32 each in DRAM, replicated), read on device each call (traceable: replays take the values of the moment). The
// chunk is in main's KV-pad-aware rotated layout over the SP rows (moe_padding_config): a row's real rows are a prefix,
// TP chip p owns its rows p S ..; padding tokens get no routes, so the dispatch, the routed experts and the combine
// never see them (their combined rows are zero). Absent: every row is real (the programs are the unpadded ones). split:
// the piece of the chunk this call runs when MoEBlock runs it as MoEBlockConfig.chunk_splits pieces (x: the piece's
// rows of each SP row; a padded chunk's real rows offset by it). The head runs the call's rows as its planner's row
// blocks (MoEHeadOutputs.row_blocks), one program whose kernels block over them. metadata: a padded forward as the
// forward's metadata words
// (common/prefill/prefill_chunk.py: at MoEBlockConfig.forward_chunks 1 (slot, start, end), above {num_chunks, 0} +
// forward_chunks records), one uint32 DRAM tensor read on device every call - in place of actual_start / actual_end;
// required for a packed forward.
MoEHeadOutputs moe_head(
    const Tensor& x,
    const MoEHeadWeights& weights,
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    std::optional<int64_t> l1_free = std::nullopt,
    const std::optional<Tensor>& arena = std::nullopt,
    const std::optional<Tensor>& actual_start = std::nullopt,
    const std::optional<Tensor>& actual_end = std::nullopt,
    uint32_t split = 0,
    const std::optional<Tensor>& metadata = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block::moe_head
