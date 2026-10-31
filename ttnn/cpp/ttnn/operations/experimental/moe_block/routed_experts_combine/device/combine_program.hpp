// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse combine's half of the experts + combine program (sparse_combine_rf.sparse_combine_rf_prepare): its cells
// (fabric ports, accumulators), kernels, circular buffers and runtime args on one chip's ProgramDescriptor, next to the
// flat routed expert's.
//
// Arena. The combine's per-core state, landing pools and local rings live in the program's arena CB (kCbArena), which
// must be the FIRST circular buffer of the program on every core (allocated before any other): it then starts at the
// same L1 address A0 on every core of every chip, which the multi-hop fabric writes (rows into a home accumulator's
// landing ring, credits and ring layouts into a port's state) rely on. Its size differs per role (port, accumulator);
// every address the combine and the flat expert's y writers exchange is an offset from A0. The port's injection ring
// (kCbInj) is the SECOND CB on the port cores, so it sits at A0 + the port arena's size (HandoffArgs carry that
// offset). Call add_combine_arena_cbs() first, then the flat expert's builder, then add_combine_program().
//
// Zeroing. The arena is program-scoped: every call starts by zeroing the combine's state on its own cores, ordered by a
// per-chip gate (kernels/dataflow/combine_zero_gate.hpp) and the senders' cross-chip barrier. The flat expert's y
// writers take part: each zeroes its words, then counts in on the gate (HandoffArgs::gate_*), before it reads them.

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "combine_host.hpp"
#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"

namespace tt::tt_metal {
class Buffer;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// Circular buffer indices on the combine's cores (disjoint from the flat expert's cells; the arena index is shared with
// the flat expert's arena on its cores).
inline constexpr uint8_t kCbInj = 0, kCbRow = 1, kCbMeta = 2, kCbTab = 3, kCbAcc = 4, kCbOut = 5, kCbW = 6,
                         kCbArena = 7;

// What the combine is planned from (k3 TtK3TokenSplitMoE.__init__: combine_auto on the flat plan's free cells).
struct CombinePlanInputs {
    uint32_t S = 0;                              // home tokens a chip
    uint32_t W = 0;                              // row width (bf16 elements): the routed width Hr
    std::vector<tt::tt_metal::CoreCoord> cells;  // logical cells the flat plan leaves free (row-major)
    uint32_t arena_bytes = 0;                    // the combine's L1 budget a core (exchange_arena_bytes)
    std::optional<uint32_t> n_acc;               // the accumulator count the fused head planned (pins A)
    std::optional<CombineMode> mode;             // ... and its mode
    uint32_t depth = 8;
    std::optional<uint32_t> piece_max;  // default max(7168, 2 W): one piece a row (the L1 hand-off)
};

// Everything the combine's program needs that depends only on the mesh, the config and the plan inputs: computed once
// per resources (MoEBlockResources::plan) and shared by every call.
struct CombinePlan {
    MeshTopology topo;
    MeshPlan mesh;  // ports: peers and fabric link ids
    uint32_t L = 0;
    FabricTieRule tie = FabricTieRule::NorthEast;
    CombineTuning tuning;
    uint32_t snap_bits = 0;
    uint32_t S = 0, W = 0;
    RowPieces pc;
    StateLayout st;
    uint32_t A = 0, MAXT = 0, PSL = 0, D = 0, QP = 0, QL = 0;
    bool fused = false;    // acc_on_ports
    bool perchip = false;  // per-chip accumulator roles (ports apart): the senders' DRAM LUT
    CreditPolicy cred;
    uint32_t far = 0, la = 0;
    CombineAutoChoice choice;

    // Arena layout (arena-relative offsets) per role.
    uint32_t port_arena = 0;  // a port core's arena CB bytes (fused: = acc_arena)
    uint32_t acc_arena = 0;   // an accumulator core's arena CB bytes
    uint32_t lut_off = 0, lut_bytes = 0;
    uint32_t loc_off = 0;  // the accumulators' local ring
    uint32_t inj_off = 0;  // the port's injection ring (= port_arena: the CB right after the arena)

    PortCores port_core;                                             // (chip, dir, link) -> logical core
    std::map<ChipCoord, std::vector<tt::tt_metal::CoreCoord>> accs;  // per chip, accumulator o's logical cell
    tt::tt_metal::CoreCoord collector;                               // accumulator 0 (every chip): the gate
    std::vector<tt::tt_metal::CoreCoord> acc_cells_union;

    // Device-derived (virtual coordinates, fabric ids), filled by make_combine_plan.
    std::vector<uint32_t> chip_ids;  // per linear chip index
    uint32_t mesh_id = 0;
    std::vector<uint16_t> lut;  // [N * A] virtual x | y << 8 of each home chip's accumulator o (perchip)
};

std::shared_ptr<const CombinePlan> make_combine_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, MoEBlockResources& resources, const CombinePlanInputs& in);

// The flat expert's side of the L1 hand-off (from its plan).
struct FlatHandoff {
    std::vector<tt::tt_metal::CoreCoord> writers;  // the y writers (logical), in the flat program's order
    uint32_t words_offset = 0;                     // their words (grants, bases, table page, stages), arena-relative
};

// The y writers' hand-off args (se_yrm.hpp SE_Y_L1; flat_routed_expert_with_program l1_args), per chip:
//   [nd, slot stride, table (buffer), words, writer index (filled by the flat builder), 3 spare,
//    per destination: noc x, noc y, slot base, slots, landed counter array]
// words, slot base and landed are ARENA-RELATIVE (the writer adds its arena base A0). Then the writers' zero gate.
struct HandoffArgs {
    std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> l1_args;
    tt::tt_metal::CoreCoord collector;  // logical: the flat writers' per-expert counts (cmb_rt)
    uint32_t counts_offset = 0;         // arena-relative (state CNT)
    tt::tt_metal::CoreCoord gate;       // virtual: count in at bar + 4 * BAR_ZEROED after zeroing the words
};

struct CombineBuffers {
    tt::tt_metal::Buffer* plan = nullptr;    // the device plan [1, 1024] uint32 (DRAM)
    tt::tt_metal::Buffer* table = nullptr;   // the hand-off table [rows_cap / 32, 64] uint32 (DRAM)
    tt::tt_metal::Buffer* out = nullptr;     // y_own [1, 1, S, W] bf16 row-major (DRAM)
    tt::tt_metal::Buffer* bar = nullptr;     // persistent words, 16 a core, every worker core
    tt::tt_metal::Buffer* hcache = nullptr;  // the senders' routed headers [8, page] uint32 (DRAM)
    tt::tt_metal::Buffer* lut = nullptr;     // per-chip roles: the accumulator LUT (DRAM)
};

// Step 1 (before any other CB of the program): the arena CB on this chip's combine cores, then the ports' injection
// rings.
void add_combine_arena_cbs(tt::tt_metal::ProgramDescriptor& desc, const CombinePlan& plan, ChipCoord chip);

// Step 2 (after the flat expert's builder): the combine's other CBs, kernels and runtime args on this chip.
HandoffArgs add_combine_program(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombinePlan& plan,
    ChipCoord chip,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& buffers,
    const FlatHandoff& flat);

// The sender's DRAM header cache page bytes (N x NH routed headers + a 64 B key) and the LUT bytes.
inline uint32_t hcache_page_bytes(uint32_t N) { return N * kNumHeaders * kHeaderStride + 64; }

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
