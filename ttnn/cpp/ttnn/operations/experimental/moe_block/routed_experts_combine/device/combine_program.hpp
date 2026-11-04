// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse combine's half of the experts + combine program: its cells
// (fabric ports, accumulators), kernels, circular buffers and runtime args on one chip's Program, next to the flat
// routed expert's (flat_routed_expert/device/flat_program.hpp) on the cells the flat plan leaves free.
//
// Arena. The combine's per-core state, landing pools, rings and CBs live in the call's arena: an L1 tensor sharded
// over the whole worker grid (the flat expert's arena too, on its cells), so at the same address on every core of
// every chip - the multi-hop fabric writes (rows into a home accumulator's landing ring, credits and ring layouts into
// a port's state) rely on it. Its first arena_bytes a core are the combine's (the reference's layout, offset for
// offset), then kWriterWordsRows x 64 B of the flat y writers' words. Every arena address in the runtime args is
// re-applied on a program-cache hit (CombineHandles::patches), the CBs are rebound to the new arena.
//
// Zeroing. The arena is allocated per call: every call starts by zeroing the combine's state on its own cores,
// ordered by a per-chip gate (kernels/dataflow/combine_zero_gate.hpp) and the senders' cross-chip barrier. The flat
// y writers take part (y_row_major.hpp SE_Y_L1): each zeroes its words, then counts in on the gate (the report's
// l1_args 5..7).

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>

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

// Circular buffer indices on the combine's cores (disjoint from the flat expert's cells; all bound to the arena).
inline constexpr uint8_t kCbInj = 0, kCbRow = 1, kCbMeta = 2, kCbTab = 3, kCbAcc = 4, kCbOut = 5, kCbW = 6;

// What the combine is planned from (k3 TtK3TokenSplitMoE.__init__: combine_auto on the flat plan's free cells).
struct CombinePlanInputs {
    uint32_t S = 0;                              // home tokens a chip
    uint32_t W = 0;                              // row width (bf16 elements): the routed width Hr
    std::vector<tt::tt_metal::CoreCoord> cells;  // logical cells the flat plan leaves free (row-major)
    uint32_t arena_bytes = 0;                    // the combine's L1 budget a core (exchange_arena_bytes)
    std::optional<uint32_t> n_acc;               // the accumulator count the fused head planned (pins A)
    std::optional<CombineMode> mode;             // ... and its mode
    uint32_t depth = 8;
};

// Everything the combine's program needs that depends only on the mesh, the config and the plan inputs: computed once
// per resources (MoEBlockResources::plan) and shared by every call.
struct CombinePlan {
    MeshTopology topo;
    MeshPlan mesh;  // ports: peers and fabric link ids
    uint32_t L = 0;
    uint32_t snap_bits = 0;
    uint32_t S = 0, W = 0;
    RowSlot pc;
    StateLayout st;
    uint32_t A = 0, MAXT = 0, PSL = 0, D = 0, QP = 0, QL = 0;
    bool fused = false;    // acc_on_ports
    bool perchip = false;  // per-chip accumulator roles (ports apart): the senders' DRAM LUT
    CreditPolicy cred;
    CombineAutoChoice choice;

    // Arena layout (arena-relative offsets; sparse_combine_rf_prepare's placement).
    uint32_t arena_bytes = 0;  // the combine's bytes a core; the y writers' words follow
    uint32_t inj_off = 0;      // the port's injection ring (ports apart: RINGS; acc_on_ports: after the local ring)
    uint32_t lut_off = 0, lut_bytes = 0;
    uint32_t tab_off_port = 0, tab_off_acc = 0;  // CB_TAB on the ports / accumulators
    uint32_t row_off = 0, meta_off = 0, acc_off = 0, out_off = 0, w_off = 0;
    uint32_t loc_off = 0;  // the accumulators' local ring

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

    bool operator==(const FlatHandoff&) const = default;
};

// The buffers the combine's runtime args address (patched on a program-cache hit).
enum class CombineBuf : uint8_t { Arena, Plan, Table, Out, Bar, Hcache, Lut, Count };
struct CombineBuffers {
    std::array<tt::tt_metal::Buffer*, static_cast<size_t>(CombineBuf::Count)> buf{};
    tt::tt_metal::Buffer* operator[](CombineBuf b) const { return buf[static_cast<size_t>(b)]; }
};

// A runtime-arg value: a literal, or a buffer's address plus an offset (re-applied on a hit).
struct CombineArg {
    uint32_t value = 0;
    std::optional<CombineBuf> buf;
};

struct CombinePatch {
    tt::tt_metal::KernelHandle kernel;
    tt::tt_metal::CoreCoord core;
    uint32_t index;
    CombineBuf buf;
    uint32_t offset;
};

inline constexpr uint32_t kUnpatched = 0xFFFFFFFFu;

inline std::array<uint32_t, static_cast<size_t>(CombineBuf::Count)> filled_unpatched() {
    std::array<uint32_t, static_cast<size_t>(CombineBuf::Count)> a;
    a.fill(kUnpatched);
    return a;
}

// What add_combine_program leaves for the caller (keep it with the cached program).
struct CombineHandles {
    std::vector<CombinePatch> patches;
    // The address each source's args were last written with (kUnpatched: unknown - the next hit writes them). A hit
    // compares the source's address with it, never with an arg's current value: the framework's inferred buffer
    // bindings (apply_resolved_bindings) may already have moved an arg that equals a buffer's address, leaving that
    // source's offset args stale.
    mutable std::array<uint32_t, static_cast<size_t>(CombineBuf::Count)> last = filled_unpatched();
    std::vector<std::pair<tt::tt_metal::CBHandle, uint32_t>> arena_cbs;  // (CB, arena byte offset)
    // The y writers' report (flat_program.hpp FlatCombineReport): the collector (virtual), its counts (arena offset),
    // and the L1 hand-off args (y_row_major.hpp SE_Y_L1): [nd, slot stride, table, words, writer index, gate x | y <<
    // 16, bar, words bytes to zero; per destination: noc x, noc y, slot base, slots, landed counter array].
    tt::tt_metal::CoreCoord collector;
    uint32_t counts_offset = 0;
    std::vector<CombineArg> l1_args;
};

// The combine's kernels, CBs and runtime args on chip `chip` of `program` (the flat expert's go on its own cells).
CombineHandles add_combine_program(
    tt::tt_metal::Program& program,
    const CombinePlan& plan,
    ChipCoord chip,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& buffers,
    const FlatHandoff& flat);

// Re-applies the buffers' addresses (a program-cache hit): a source's args (and the arena's CBs) only when its address
// moved.
void override_combine_addresses(tt::tt_metal::Program& program, const CombineHandles& h, const CombineBuffers& buffers);

// The writers' l1_args with the buffers' current addresses.
std::vector<uint32_t> resolve_l1_args(const CombineHandles& h, const CombineBuffers& buffers);

// The sender's DRAM header cache page bytes (N x NH routed headers + a 64 B key).
inline uint32_t hcache_page_bytes(uint32_t N) { return N * kNumHeaders * kHeaderStride + 64; }

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
