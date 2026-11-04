// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's programs, one a chip: the shared-expert engine, the x gather, the router and the routing exchange.
//
// HeadModel is everything the build derives from the planned layout and the mesh, once a block (the engine's sizing,
// the reduce-scatter plan, the gather's rings / ports / copy cores, the router cores and their layout, the routing
// exchange's senders, paths and planner instances, the persistent words each piece needs). build_head_workload makes
// the per-chip programs from a model, the persistent words' addresses and the tensors:
//  * the arena is an L1 tensor given to the op (by default one the block's resources keep), not one allocated each
//    call: the same address on every core and chip, so the CBs over its regions bind to it at their offsets;
//  * the persistent words (mailboxes, flags, counters) are word tensors of the block's resources;
//  * every address argument is recorded with its source and offset (HeadPatches): a cache hit rewrites only the
//    arguments of sources whose address moved;
//  * addresses are common runtime args, never defines (kernels/head_kernel_contract.hpp);
//  * the fabric connections are main's (append_fabric_connection_rt_args); the routing planner splits torus ties by
//    the source's parity, as main's fabric.

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
    bool sqrtsoftplus = false;              // router score sqrt(softplus); else sigmoid
    uint32_t handoff_bytes = 16;            // MoEBlockConfig.handoff_bytes()
    int64_t l1_free = kL1Free - kL1Margin;  // the free L1 the plan assumed (the engine sizes its arena under its cap)
    int64_t cb_floor = 0;                   // the other static CBs under the arena the plan assumed (arena_cap_of)

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

// The buffers the programs address. Every address argument is one of these plus an offset.
enum HeadSrc : uint16_t {
    SRC_X = 0,   // [rows, H / TP] this chip's shard
    SRC_X_FULL,  // the DRAM fallback's gathered x [rows, H]
    SRC_W,       // the shared expert's weights in the engine's layout
    SRC_Y,       // the shared expert's reduced output
    SRC_ROWS,    // the owners' joint routing rows
    SRC_XBUF,    // the dispatch buffer
    SRC_ROUTE,   // the owned tokens' route words
    SRC_RW,      // W_r^T [E, H / TP]
    SRC_RB,      // the router bias tiles
    SRC_RIX,     // the expert index tiles
    SRC_SLOTS,   // the routing exchange's DRAM slots
    SRC_FSLOTS,  // its forwarding slots
    SRC_COUNTS,  // the combine plan: counts, regions, token_index, table, plan
    SRC_REGIONS,
    SRC_TOKEN_INDEX,
    SRC_TABLE,
    SRC_PLAN,
    SRC_ARENA,  // the arena (L1)
    SRC_START,  // a padded chunk's actual_start (main's chunk metadata scalar: one uint32, DRAM; absent: all real)
    SRC_END,    // its actual_end
    SRC_WORD0,  // the persistent words, in head_words order
};

inline constexpr size_t kHeadMaxSrc = 40;  // SRC_WORD0 + the persistent words, at most

struct HeadBuffers {
    // per HeadSrc (SRC_WORD0 + i: head_words' i-th): the buffer (nullptr for the words, an absent tensor) and its
    // address (0 when absent). Fixed arrays: a cache hit builds one a chip.
    std::array<tt::tt_metal::Buffer*, kHeadMaxSrc> buffers{};
    std::array<uint32_t, kHeadMaxSrc> addresses{};
    tt::tt_metal::Buffer* operator[](HeadSrc s) const { return buffers[s]; }
};

// Where every address argument of one chip's program sits: a cache hit compares each source's address with the one
// the program holds and rewrites that source's arguments only when it moved.
struct HeadPatches {
    struct Arg {
        uint32_t kernel;
        bool common;
        CoreCoord core;
        uint32_t index;
        uint16_t src;
        uint32_t offset;
    };
    struct Cb {
        uint32_t index;  // in the program's CBs (descriptor order)
        uint32_t offset;
    };
    std::vector<Arg> args;  // grouped by source
    // [begin, end) of each source's args (one entry a source present): a hit compares one address a source, never
    // walks the args of a source that did not move
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    std::vector<Cb> arena_cbs;
};

// The address each source's args of one cached program were last written with (0xFFFFFFFF: unknown - the next hit
// writes them). A hit compares a source's address with it, never with an arg's current value: the framework's
// inferred buffer bindings (apply_resolved_bindings) may already have moved an arg that equals a buffer's address,
// leaving that source's offset args stale.
struct HeadLastAddresses {
    std::array<uint32_t, kHeadMaxSrc> a = [] {
        std::array<uint32_t, kHeadMaxSrc> v;
        v.fill(0xFFFFFFFFu);
        return v;
    }();
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
    uint32_t slots_rows, fslots_rows, slot_words;            // uint32 ROW_MAJOR [rows, slot_words]
    std::array<std::pair<uint32_t, uint32_t>, 5> plan_outs;  // uint32 ROW_MAJOR
    uint32_t arena_bytes;
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

// The cores the arena spans (the matmul cores, the router cores, the ports, the watcher) and its bytes a core.
CoreRangeSet head_arena_cores(const HeadModel& model);

// The per-chip programs (one per mesh coordinate of `coords`). patches: per coordinate (r C + c) where the address
// arguments sit.
tt::tt_metal::WorkloadDescriptor build_head_workload(
    const HeadModel& model,
    const HeadBuffers& buffers,
    const std::vector<ttnn::MeshCoordinate>& coords,
    std::vector<HeadPatches>* patches = nullptr);

// Rewrites the address arguments (and the arena CBs) of one chip's cached program whose sources moved.
void patch_head_program(
    tt::tt_metal::Program& program, const HeadPatches& patches, const HeadBuffers& buffers, HeadLastAddresses& last);

}  // namespace ttnn::operations::experimental::moe_block::fused_head
