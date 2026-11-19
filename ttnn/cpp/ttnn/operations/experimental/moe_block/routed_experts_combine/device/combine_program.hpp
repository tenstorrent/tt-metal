// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse combine's half of the experts + combine program: its cores
// (fabric ports, accumulators), kernels, circular buffers and runtime args on one chip's Program, next to the flat
// routed expert's (flat_routed_expert/device/flat_program.hpp) on the cores the flat plan leaves free.
//
// Arena. The combine's per-core state, landing pools, rings and CBs live in the call's arena: an L1 tensor sharded
// over the whole worker grid (the flat expert's arena too, on its cores), so at the same address on every core of
// every chip - the multi-hop fabric writes (rows into a home accumulator's landing ring, credits and ring layouts into
// a port's state) rely on it. Its first arena_bytes a core are the combine's (the reference's layout, offset for
// offset), then kWriterWordsRows x 64 B of the flat y writers' words. Every address arg is recorded with its source
// (CombineHandles::patches): a fixup of the args table, whose sources - and the arena's CBs - every launch rewrites.
//
// Zeroing. The arena is allocated per call: every call starts by zeroing the combine's state on its own cores,
// ordered by a per-chip gate (kernels/dataflow/combine_zero_gate.hpp) and the senders' cross-chip barrier. The flat
// y writers take part (y_row_major.hpp combine_handoff): each zeroes its words, then counts in on the gate (the
// report's l1_args 5..7).

#pragma once
#include <tuple>

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

// Circular buffer indices on the combine's cores (disjoint from the flat expert's cores; all bound to the arena).
inline constexpr uint8_t kCbInj = 0, kCbRow = 1, kCbMeta = 2, kCbTab = 3, kCbAcc = 4, kCbOut = 5, kCbW = 6;

// What the combine is planned from (k3 TtK3TokenSplitMoE.__init__: combine_auto on the flat plan's free cores).
struct CombinePlanInputs {
    uint32_t home_tokens = 0;                         // home tokens a chip (S)
    uint32_t row_elements = 0;                        // row width (bf16 elements): the routed width Hr (W)
    std::vector<tt::tt_metal::CoreCoord> free_cores;  // logical cores the flat plan leaves free (row-major)
    uint32_t arena_bytes = 0;                         // the combine's L1 budget a core (exchange_arena_bytes)
    std::optional<uint32_t> accumulator_count;        // the accumulator count the MoE head planned (pins A)
    std::optional<CombineMode> mode;                  // ... and its mode
    uint32_t depth = 8;
    std::optional<CombineLanding> landing;  // nullopt: rings
    std::optional<uint32_t> credit_batch;   // nullopt: the planner's (credit_policy)
    // CombineLanding::FlatCores: the arena the flat expert leaves free on its cores (FlatRoutedExpertPlan::free_l1):
    // cores with the same free spans ([begin, end) offsets)
    struct FlatFree {
        std::vector<tt::tt_metal::CoreCoord> cores;
        std::vector<std::pair<uint32_t, uint32_t>> spans;
    };
    std::vector<FlatFree> flat_free;
};

// Everything the combine's program needs that depends only on the mesh, the config and the plan inputs: computed once
// per resources (MoEBlockResources::plan) and shared by every call.
struct CombinePlan {
    MeshTopology topo;
    MeshPlan mesh;              // ports: peers and fabric link ids
    FirstHops hops;             // the fabric's first hop between every pair of chips: the port a row / credit leaves by
    uint32_t fabric_links = 0;  // fabric links between neighbor chips (L)
    uint32_t snap_bits = 0;
    uint32_t home_tokens = 0;   // home tokens a chip (S)
    uint32_t row_elements = 0;  // row width, bf16 elements (W)
    RowSlot row_slot;
    StateLayout state;
    uint32_t accumulator_count = 0;              // accumulators a chip (A)
    uint32_t max_tokens_per_accumulator = 0;     // home tokens an accumulator sums at most (ceil(S / A))
    uint32_t landing_slots_per_accumulator = 0;  // the landing pool's slots an accumulator (PSL)
    uint32_t landing_slots_per_source = 0;       // ... a source chip's share, floored (D)
    uint32_t injection_ring_slots = 0;           // a port's injection ring (QP)
    uint32_t local_ring_slots = 0;               // an accumulator's local ring (QL)
    bool accumulators_on_ports = false;          // the accumulators on the ports (CombineMode::AccOnPorts)
    bool per_chip_accumulator_table = false;     // per-chip accumulator roles (ports apart): the senders' DRAM table
    // (ports apart) every core that is a port or an accumulator on some chip: each runs one binary a RISC with both
    // roles and picks this chip's (combine_port_or_accumulator_reader / _writer.cpp) - the same program on every chip,
    // one program for the mesh (common/chip_program.hpp)
    std::vector<tt::tt_metal::CoreCoord> port_or_accumulator_cores;
    bool shared_landing = false;  // the shared landing pool (CombineLanding::Shared)
    // CombineLanding::FlatCores: accumulator o's ring slot i is pool slot o x landing_slots_per_accumulator + i of the
    // flat expert cores' free arena, pool_args its description (kernels/dataflow/combine_pool.hpp; the same cores on
    // every chip), pool_slots its slots
    bool landing_on_flat_expert_cores = false;
    uint32_t pool_slots = 0;
    uint32_t row_depth = 2;  // the accumulators' row / meta / weight CBs (rows): kPoolRows on the flat expert cores
    uint32_t accumulator_table_cb_bytes = kTabCbBytes;  // the accumulators' CB_TAB bytes (acc_tab_bytes)
    std::vector<uint32_t> pool_args;
    CreditPolicy credit_policy;
    CombineAutoChoice choice;

    // Arena layout (arena-relative offsets; sparse_combine_rf_prepare's placement).
    uint32_t arena_bytes = 0;  // the combine's bytes a core; the y writers' words follow
    // the port's injection ring (ports apart: StateLayout::rings_offset; accumulators on the ports: past the local
    // ring)
    uint32_t injection_ring_offset = 0;
    uint32_t accumulator_table_offset = 0, accumulator_table_bytes = 0;  // the per-chip accumulator table's staging
    uint32_t port_table_cb_offset = 0, accumulator_table_cb_offset = 0;  // CB_TAB on the ports / accumulators
    // the accumulators' CBs: rows (CB_ROW), meta (CB_META), the fp32 sums (CB_ACC), output (CB_OUT), weights (CB_W)
    uint32_t row_cb_offset = 0, meta_cb_offset = 0, accumulator_fp32_offset = 0, output_cb_offset = 0,
             weight_cb_offset = 0;
    uint32_t local_ring_offset = 0;  // the accumulators' local ring

    PortCores port_core;  // (chip, dir, link) -> logical core
    PortLuts port_luts;   // the chips' port ids (MoEBlockResources::port_luts: the MoE head numbers them alike)
    std::map<ChipCoord, std::vector<tt::tt_metal::CoreCoord>> accumulator_cores;  // per chip, accumulator o's core
    tt::tt_metal::CoreCoord collector;  // accumulator 0 (every chip): the gate
    std::vector<tt::tt_metal::CoreCoord> accumulator_cores_union;

    // Device-derived (virtual coordinates, fabric ids), filled by make_combine_plan.
    std::vector<uint32_t> chip_ids;  // per linear chip index
    uint32_t mesh_id = 0;
    // [N * A] virtual x | y << 8 of each home chip's accumulator o (per_chip_accumulator_table)
    std::vector<uint16_t> accumulator_table;
};

std::shared_ptr<const CombinePlan> make_combine_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, MoEBlockResources& resources, const CombinePlanInputs& in);

// The flat expert's side of the L1 hand-off (from its plan).
struct FlatHandoff {
    std::vector<tt::tt_metal::CoreCoord> writers;  // the y writers (logical), in the flat program's order

    bool operator==(const FlatHandoff&) const = default;
};

// The buffers the combine's runtime args address (rewritten on every launch).
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

// A role block of a port-or-accumulator core that is not this chip's role (combine_port_or_accumulator_reader.cpp): the
// port reader's arg 4 / the sender's arg 3 is this (never run - the block holds its buffers at a real block's args).
inline constexpr uint32_t kIdlePort = 0xFFFFFFFFu;
// the port reader's table CB when the core carries both roles (the accumulators' kCbTab sits elsewhere in the arena)
inline constexpr uint8_t kCbTabP = 8;

// What add_combine_program leaves for the caller. Build-time only: the address args (source, offset) and the report
// placement, which the caller turns into args-table fixups. Not kept with the cached program.
struct CombineHandles {
    std::vector<CombinePatch> patches;  // every address arg (the descriptor holds the whole address)
    // every sender's fabric connection args (kernel, core, first, count): they stay runtime args (args_table.hpp, kept
    // args), which the kernel opens with build_from_args
    struct FabricArgs {
        tt::tt_metal::KernelHandle kernel;
        tt::tt_metal::CoreCoord core;
        uint32_t first;
        uint32_t count;
    };
    std::vector<FabricArgs> fabric_args;
    // The y writers' report (flat_program.hpp FlatCombineReport): the collector (virtual), its counts (arena offset),
    // and the L1 hand-off args (y_row_major.hpp combine_handoff): [nd, slot stride, table, words, writer index, gate x
    // | y << 16, bar, words bytes to zero; per destination: noc x, noc y, slot base, slots, landed counter array].
    tt::tt_metal::CoreCoord collector;
    uint32_t counts_offset = 0;
    std::vector<CombineArg> l1_args;
};

// The combine's kernels, CBs and runtime args on chip `ch` of `program`, a program descriptor (the flat expert's go
// on its own cores). Its CBs are backed by the arena buffer (common/args_table.hpp arena_backed_cbs finds them).
CombineHandles add_combine_program(
    tt::tt_metal::ProgramDescriptor& program,
    const CombinePlan& plan,
    ChipCoord ch,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& bufs,
    const FlatHandoff& flat);

// The writers' l1_args with the buffers' (`bufs`) current addresses.
std::vector<uint32_t> resolve_l1_args(const CombineHandles& h, const CombineBuffers& bufs);

// The sender's DRAM header cache page bytes (N x NH routed headers + a 64 B key).
inline uint32_t hcache_page_bytes(uint32_t N) { return N * kNumHeaders * kHeaderStride + 64; }

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
