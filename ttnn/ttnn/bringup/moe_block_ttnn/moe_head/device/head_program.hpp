// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's program, one for the mesh: the shared-expert engine, the x gather, the router and the routing
// exchange.
//
// HeadModel is everything the build derives from the planned layout and the mesh, once a block (the engine's sizing,
// the reduce-scatter plan, the gather's rings / ports / copy cores, the router cores and their layout, the routing
// exchange's senders, paths and planner instances, the persistent words each piece needs). build_head_workload makes
// the per-chip programs from a model, the persistent words' addresses and the tensors:
//  * the arena is an L1 tensor given to the op (by default one the block's resources keep), not one allocated each
//    call: the same address on every core and chip, so the CBs over its regions bind to it at their offsets;
//  * the persistent words (mailboxes, flags, counters) are word tensors of the block's resources;
//  * every address argument is recorded with its source and offset (HeadAddressArgs); every kernel's args then go to
//    per-chip DRAM tables (head_args_to_dram: an address arg as its offset plus its source's address, a common arg the
//    kernel holds), so the chips' programs are alike and one program serves the mesh - a cache hit rewrites the
//    sources' addresses in the kernels' common args (common/args_table.hpp write_program_addresses);
//  * addresses are common runtime args, never defines (README.md, Kernels);
//  * the fabric connections are main's (append_fabric_connection_rt_args, opened with build_from_args: their args stay
//    runtime args, kept by the args table - the eth channel a per-device runtime arg); the routing exchange and planner
//    route by the fabric's own first hops (HeadShape.first_hops: asked of the fabric).

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
#include <ttnn/per_device_program_merge.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "../../common/args_table.hpp"
#include "../../common/moe_block_config.hpp"
#include "../../common/mesh_routing.hpp"

#include "head_plan.hpp"

namespace tt::tt_metal {
class Buffer;
}
namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::bringup::moe_block::moe_head {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

// Everything structural the head is built from (hashed): the planned layout, the model, the routing exchange, the
// config switches the kernels see.
struct HeadProgramConfig {
    HeadShape shape;
    HeadLayout layout;
    ExpertActivation activation;
    tt::tt_metal::MathFidelity fidelity = tt::tt_metal::MathFidelity::HiFi2;
    tt::tt_metal::DataType w_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    float route_scale = 1.0f;
    bool sqrt_softplus = false;             // router score sqrt(softplus); else sigmoid
    uint32_t handoff_bytes = 16;            // MoEBlockConfig.handoff_bytes()
    int64_t l1_free = kL1Free - kL1Margin;  // the free L1 the plan assumed (the engine sizes its arena under its cap)
    int64_t cb_floor = 0;                   // the other static CBs under the arena the plan assumed (arena_cap_of)

    bool operator==(const HeadProgramConfig&) const = default;
};

// A persistent word tensor the head needs: name (resources slot "moe_head/<name>"), its cores, uint32 words a core.
struct HeadWordsSpec {
    std::string name;
    CoreRangeSet cores;
    uint32_t words;
};

// The buffers the programs address. Every address argument is one of these plus an offset.
enum HeadSource : uint16_t {
    SOURCE_X = 0,            // [rows, H / TP] this chip's shard
    SOURCE_W,                // the shared expert's weights in the engine's layout
    SOURCE_Y,                // the shared expert's reduced output
    SOURCE_ROWS,             // the owners' joint routing rows
    SOURCE_DISPATCH_BUFFER,  // the dispatch buffer
    SOURCE_ROUTE,            // the owned tokens' route words
    SOURCE_ROUTER_W,         // W_r^T [E, H / TP]
    SOURCE_ROUTER_BIAS,      // the router bias tiles
    SOURCE_ROUTER_INDEX,     // the expert index tiles
    SOURCE_SLOTS,            // the routing exchange's DRAM slots
    SOURCE_COUNTS,           // the combine plan: counts, regions, token_index, table, plan
    SOURCE_REGIONS,
    SOURCE_TOKEN_INDEX,
    SOURCE_TABLE,
    SOURCE_PLAN,
    SOURCE_ARENA,  // the arena (L1)
    SOURCE_START,  // a padded chunk's actual_start (main's chunk metadata scalar: one uint32, DRAM; absent: all real)
    SOURCE_END,    // its actual_end
    SOURCE_FIRST_WORD,  // the persistent words, in head_words order
};

inline constexpr size_t kHeadMaxSources = 40;  // SOURCE_FIRST_WORD + the persistent words, at most
// row blocks in one program (kernels/row_block.hpp): a core's block words - 5 done words, the go word, the
// coordinator's count, 16 B apart (kernels/head_shared_words.hpp kWordsBytes; head_row_blocks.cpp checks it)
inline constexpr uint32_t kHeadRowBlockWordsBytes = 112;

struct HeadBuffers {
    // per HeadSource (SOURCE_FIRST_WORD + i: head_words' i-th): the buffer (nullptr for the words, an absent tensor)
    // and its address (0 when absent). Fixed arrays: a cache hit builds one a chip.
    std::array<tt::tt_metal::Buffer*, kHeadMaxSources> buffers{};
    std::array<uint32_t, kHeadMaxSources> addresses{};
    // row blocks (the head's row blocks k): k, this call's block of its split, its split (chunk_splits)
    uint32_t row_blocks = 1, block = 0, split = 0;
    // row blocks in one program (merged, k > 1): the call runs every block of its split - x and y hold them
    // all, block l's at l x stride a bank (B) further; the block words' address (kernels/row_block.hpp)
    bool merged = false;
    // a padded forward given as the forward's metadata words (moe_head's metadata=, at SOURCE_START; SOURCE_END absent)
    bool padded_forward_metadata = false;
    uint32_t x_stride = 0, y_stride = 0, row_block_words = 0;
    tt::tt_metal::Buffer* operator[](HeadSource source) const { return buffers[source]; }
};

// Where every address argument of one chip's program sits (ChipProgram::build: the input of head_args_to_dram).
struct HeadAddressArgs {
    struct Arg {
        uint32_t kernel;
        bool common;
        CoreCoord core;
        uint32_t index;
        uint16_t source;
        uint32_t offset;
        // not an address: the first of a fabric connection's args (kFabricConnectionArgs from `index`), which stay
        // runtime args (args_table.hpp, kept args) for build_from_args - their eth channel differs by chip
        bool fabric_connection = false;
    };
    std::vector<Arg> args;
};

class HeadModel;

// The host model of the head on this mesh (computed once a block; heavy).
std::shared_ptr<const HeadModel> make_head_model(
    tt::tt_metal::distributed::MeshDevice& mesh, const HeadProgramConfig& config, const PortLuts& luts);

// The persistent words the model's programs need (create them in the resources before building).
std::vector<HeadWordsSpec> head_words(const HeadModel& model);

// Shapes of the head's DRAM tensors (rows of uint32 / elements; see moe_head.cpp for the specs).
struct HeadTensorShapes {
    uint32_t y_rows, y_columns;                                 // bf16 TILE
    uint32_t routing_rows_rows, routing_rows_columns;           // uint16 ROW_MAJOR
    uint32_t dispatch_buffer_rows, dispatch_buffer_columns;     // bf16 ROW_MAJOR
    uint32_t route_words;                                       // uint32 ROW_MAJOR [1, route_words]
    uint32_t slots_rows, slot_words;                            // uint32 ROW_MAJOR [rows, slot_words]
    std::array<std::pair<uint32_t, uint32_t>, 5> plan_outputs;  // uint32 ROW_MAJOR
    uint32_t arena_bytes;
};
HeadTensorShapes head_tensor_shapes(const HeadModel& model);

// The shared expert's and the router's weights in the head's layouts: host row-major floats - gate / up [Is, H],
// down [H, Is], latent_down [Hr, H] (or empty), router_w [E, H], router_b [E].
struct HeadDeviceWeights {
    Tensor w, router_w, router_bias, router_index;
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

// The head's workload over `coords`: every chip's program built, their args moved to per-chip DRAM tables, then one
// program for the whole mesh (a program a chip, logged, only if the chips' programs differ in structure). addresses:
// per program (in workload.programs order), what a cache hit writes (and the DRAM tables, kept with it).
struct HeadWorkload {
    tt::tt_metal::WorkloadDescriptor workload;
    std::vector<ProgramAddresses> addresses;
    // per program: its runtime-arg words that differ by chip (the fabric connections' eth channels; empty: none)
    std::vector<std::vector<ttnn::PerDeviceRuntimeArgsField>> per_device_runtime_args;
};
HeadWorkload build_head_workload(
    const HeadModel& model, const HeadBuffers& buffers, const std::vector<ttnn::MeshCoordinate>& coords);

}  // namespace ttnn::operations::bringup::moe_block::moe_head
