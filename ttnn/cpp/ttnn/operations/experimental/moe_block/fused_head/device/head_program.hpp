// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's programs, one a chip (reference: models/demos/moe_block/tt/fused_head.py HeadSE3._run and the
// pieces it assembles - se_engine3.SeEngine3.pieces, x_gather3.XGather3.pieces, router_stream.RouterStream.pieces,
// routing_stream.RoutingStream - on mstaletovic/moe-trim).
//
// HeadModel is everything the build derives from the planned layout and the mesh, once a block (the engine's sizing,
// the reduce-scatter plan, the gather's rings / ports / copy cores, the router cores and their layout, the routing
// exchange's senders, paths and planner instances, the persistent words each piece needs). build_head_workload makes
// the per-chip programs from a model, the persistent words' addresses and the tensors. Differences from the
// reference, none in what the kernels compute or in their NoC / fabric traffic:
//  * no per-call arena tensor: the arena is a program CB, the first CB on every core it spans, so it sits at the L1
//    CB base on every core and chip; the CBs over its regions are bound to a view of that range (a non-owning L1
//    buffer at the base), and every arena address a kernel takes is base + the region's offset (a constant);
//  * the persistent words (mailboxes, flags, counters, the reference's global semaphores) are word tensors of the
//    block's resources;
//  * addresses the reference baked into defines are common runtime args (kernels/head_kernel_contract.hpp);
//  * the fabric connections are main's (append_fabric_connection_rt_args); the routing planner's torus tie split is
//    the config's tie rule (main's fabric: NorthEast).

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"

#include "head_plan.hpp"

namespace tt::tt_metal {
class Buffer;
}
namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::fused_head {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

// Everything structural the head is built from (hashed): the planned layout, the model, the routing exchange, the
// config switches the kernels see.
struct HeadProgramConfig {
    HeadShape shape;
    HeadLayout lay;
    ExpertActivation glu;
    tt::tt_metal::MathFidelity fidelity = tt::tt_metal::MathFidelity::HiFi2;
    tt::tt_metal::DataType w_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    float route_scale = 1.0f;
    bool sqrtsoftplus = false;  // router score sqrt(softplus); else sigmoid
    bool ordered = true;        // MoEBlockConfig.deterministic
    uint32_t rows_cap = 0;      // the plan's row capacity (0: the worst case)
    FabricTieRule tie_rule = FabricTieRule::NorthEast;
    uint32_t handoff_bytes = 16;            // MoEBlockConfig.handoff_bytes()
    int64_t l1_free = kL1Free - kL1Margin;  // the free L1 the plan assumed (the engine sizes its arena under its cap)

    bool operator==(const HeadProgramConfig&) const = default;
};

// A persistent word tensor the head needs: name (resources slot "fused_head/<name>"), its cores, uint32 words a core.
struct HeadWordsSpec {
    std::string name;
    CoreRangeSet cores;
    uint32_t words;
};

// The persistent words' base addresses (the same on every chip), by HeadWordsSpec name.
using HeadWordAddresses = std::map<std::string, uint32_t>;

// The DRAM tensors the programs address (their buffers; bound on cache hits).
struct HeadBuffers {
    tt::tt_metal::Buffer* x = nullptr;                 // [rows, H / TP] this chip's shard
    tt::tt_metal::Buffer* x_full = nullptr;            // the DRAM fallback's gathered x [rows, H]
    tt::tt_metal::Buffer* w = nullptr;                 // the shared expert's weights in the engine's layout
    tt::tt_metal::Buffer* y = nullptr;                 // the shared expert's reduced output
    tt::tt_metal::Buffer* rows = nullptr;              // the owners' joint routing rows
    tt::tt_metal::Buffer* xbuf = nullptr;              // the dispatch buffer
    tt::tt_metal::Buffer* route = nullptr;             // the owned tokens' route words
    tt::tt_metal::Buffer* latent = nullptr;            // the owners' latent rows
    tt::tt_metal::Buffer* rw = nullptr;                // W_r^T [E, H / TP]
    tt::tt_metal::Buffer* rb = nullptr;                // the router bias tiles
    tt::tt_metal::Buffer* rix = nullptr;               // the expert index tiles
    tt::tt_metal::Buffer* slots = nullptr;             // the routing exchange's DRAM slots
    tt::tt_metal::Buffer* fslots = nullptr;            // its forwarding slots
    std::array<tt::tt_metal::Buffer*, 5> plan_outs{};  // counts, regions, token_index, table, plan
};

class HeadModel;

// The host model of the head on this mesh (computed once a block; heavy).
std::shared_ptr<const HeadModel> make_head_model(
    tt::tt_metal::distributed::MeshDevice& mesh, const HeadProgramConfig& config, const PortLuts& luts);

// The persistent words the model's programs need (create them in the resources before building).
std::vector<HeadWordsSpec> head_words(const HeadModel& model);

// Shapes of the head's DRAM tensors (rows of uint32 / elements; see fused_head.cpp for the specs).
struct HeadTensorShapes {
    uint32_t y_rows, y_cols;                                 // bf16 TILE
    uint32_t rows_rows, rows_cols;                           // uint16 ROW_MAJOR
    uint32_t xbuf_rows, xbuf_cols;                           // bf16 ROW_MAJOR
    uint32_t route_words;                                    // uint32 ROW_MAJOR [1, route_words]
    uint32_t latent_rows, latent_cols;                       // bf16 TILE (0: no latent)
    uint32_t slots_rows, fslots_rows, slot_words;            // uint32 ROW_MAJOR [rows, slot_words]
    std::array<std::pair<uint32_t, uint32_t>, 5> plan_outs;  // uint32 ROW_MAJOR
    uint32_t w_tiles_per_region, w_regions;                  // the engine weights: regions of tiles (bank-sharded)
    uint32_t arena_bytes;
    uint32_t arena_base;  // the L1 CB base (the arena CB's address)
};
HeadTensorShapes head_tensor_shapes(const HeadModel& model);

// The shared expert's and the router's weights in the head's layouts (SeEngine3.upload, RouterStream.upload): host
// row-major floats - gate / up [Is, H], down [H, Is], latent_down [Hr, H] (or empty), router_w [E, H], router_b [E].
struct HeadDeviceWeights {
    Tensor w, rw, rb, rix;
};
HeadDeviceWeights prepare_head_weights(
    const HeadModel& model,
    const std::vector<float>& gate,
    const std::vector<float>& up,
    const std::vector<float>& down,
    const std::vector<float>& latent_down,
    const std::vector<float>& router_w,
    const std::vector<float>& router_b);

// The per-chip programs (one per mesh coordinate of `coords`). fabric_starts (tests): "r,c,kernel file,x,y" -> the
// index the core's fabric connection args start at.
tt::tt_metal::WorkloadDescriptor build_head_workload(
    const HeadModel& model,
    const HeadWordAddresses& words,
    const HeadBuffers& buffers,
    const std::vector<ttnn::MeshCoordinate>& coords,
    std::map<std::string, uint32_t>* fabric_starts = nullptr);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
