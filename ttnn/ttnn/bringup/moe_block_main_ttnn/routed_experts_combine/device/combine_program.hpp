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
#include <tt-metalium/runtime_args_data.hpp>

#include "ttnn/tensor/tensor.hpp"

#include "combine_host.hpp"
#include "../../common/mesh_routing.hpp"
#include "../../common/port_placement.hpp"

namespace tt::tt_metal {
class Buffer;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::operations::bringup::moe_block_main {
class MoEBlockResources;
}

namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine {

// Circular buffer indices on the combine's cores (disjoint from the flat expert's cells; all bound to the arena).
inline constexpr uint8_t kCbInj = 0, kCbRow = 1, kCbMeta = 2, kCbTab = 3, kCbAcc = 4, kCbOut = 5, kCbW = 6;

// MOE_FILL=piggyback (P.53): expert replica copies carried by the combine's port senders as extra fabric packets.
// One job: chip src's port facing the first hop toward chip dst (link 0) reads pieces [p0, p1) of run table `table`
// (expert_slot_copy's tables and chunking: each run cut in expert_slot_copy_chunk_bytes pieces, numbered across the
// table's runs) from its own DRAM and writes them into chip dst's DRAM (fill.gu / fill.d: the same buffers on every
// chip), sending a piece only when no row of its own is waiting; `last`: after the job's last piece, a flushed atomic
// increment of word src of chip dst's done words (the fill's completion, cumulative).
struct CombineFillJob {
    uint32_t src = 0, dst = 0, table = 0, p0 = 0, p1 = 0;
    bool last = false;
};
inline constexpr uint32_t kFillMaxJobs = 4;                  // jobs a port a call
inline constexpr uint32_t kFillArgs = 6 + 4 * kFillMaxJobs;  // the port sender's fill args (the last ones)
inline constexpr uint32_t kFillStages = 4;                   // pieces staged in a port's L1 (read, then sent)
inline constexpr uint32_t kFillTabBytes = 16384;             // a run table's L1 copy on a port
struct CombineFillCall {
    Tensor gu, d, runs;  // the copied layer's routed weights (read on src, written on dst) and the run tables
    std::vector<CombineFillJob> jobs;
    uint32_t cap = 1;  // a piece goes only while the port's fabric channel holds fewer than cap packets (main loop)
    // MOE_REPLICAS=exact (P.55b, REC_FX): the jobs are not runtime args but pages of this uint32 DRAM tensor, per chip
    // [kFxPages, kFxPageWords] (page dir x L + link: that port's jobs; page kFxChipPage: the chip's waits; layout in
    // combine_port_writer.cpp); `jobs` and `cap` are then unused. exact_limit: the cycles a relay's gate or the
    // waiter's wait may take before it gives up (never a hang).
    std::optional<Tensor> exact;
    uint32_t exact_limit = 0;
    // P.60 (MOE_REPLICAS=prev, MOE_PREV_HOST=device): exact_ring > 0 - `exact` holds exact_ring slots of kFxPages pages
    // a chip (slot s = pages [s kFxPages, (s + 1) kFxPages)) and every port reads the call's slot itself: word 0 of
    // exact_ordinal (uint32 DRAM, page 0 in bank 0; the device planner's call ordinal) mod exact_ring. The fill args
    // then hold no per-call value, so a cache hit writes them only when they change (the factory's fill signature).
    // exact_wait false: the flat expert's weight readers do not wait for a landed copy (no token: the copies of a
    // device planner land in earlier calls). Defaults (0, nullopt, true): the behaviour before the option.
    uint32_t exact_ring = 0;
    std::optional<Tensor> exact_ordinal;
    bool exact_wait = true;
    // P.62b (DeepSeek-V4.1 MOE_PLANNER=folded; needs exact_ring): after its main loop each link-0 port sends this
    // chip's expert counts row (prev_counts: the fused head's counts, one uint32 page [chips x prev_spc]; the chip's
    // own slots) to every chip whose first hop is that port - a fused write into row (G[0] & 1) x chips + me of
    // prev_landing (uint32 DRAM, rows of al16(prev_spc) words; the last word the tag G[0] + 1) and an atomic increment
    // of the word prev_sem on core prev_core (logical) there: the replica planner's input, read by the next fused
    // head's watcher tail (HeadTail). The chip's first port also writes the row into its own landing row. Absent: no
    // packet.
    std::optional<Tensor> prev_counts, prev_landing;
    uint32_t prev_sem = 0, prev_spc = 0;
    tt::tt_metal::CoreCoord prev_core{0, 0};
};
inline constexpr uint32_t kFxPages = 9, kFxChipPage = 8, kFxPageWords = 256, kFxMaxJobs = 30, kFxWaits = 4;
inline constexpr uint32_t kFxWords = 452;  // the exact words a chip: 2 x 32 flags, 8 x 16 stamps, 4 x 64 arrivals,
                                           // the landed word (P.55c: this call's token once the copy has landed)
inline constexpr uint32_t kFxLanded = 448;
inline constexpr uint32_t kFxL1 = 3328;  // a port's L1 for the exact path past the staging slots

// What the combine is planned from (k3 TtK3TokenSplitMoE.__init__: combine_auto on the flat plan's free cells).
struct CombinePlanInputs {
    uint32_t S = 0;                              // home tokens a chip
    uint32_t W = 0;                              // row width (bf16 elements): the routed width Hr
    std::vector<tt::tt_metal::CoreCoord> cells;  // logical cells the flat plan leaves free (row-major)
    uint32_t arena_bytes = 0;                    // the combine's L1 budget a core (exchange_arena_bytes)
    std::optional<uint32_t> n_acc;               // the accumulator count the fused head planned (pins A)
    std::optional<CombineMode> mode;             // ... and its mode
    uint32_t depth = 8;
    std::optional<CombineLanding> landing;  // nullopt: rings
    std::optional<uint32_t> credit_batch;   // nullopt: the planner's (credit_policy)
    // CombineLanding::FlatCores: the arena the flat expert leaves free on its cells (FlatRoutedExpertPlan::free_l1):
    // cells with the same free spans ([begin, end) offsets)
    struct FlatFree {
        std::vector<tt::tt_metal::CoreCoord> cells;
        std::vector<std::pair<uint32_t, uint32_t>> spans;
    };
    std::vector<FlatFree> flat_free;
};

// Everything the combine's program needs that depends only on the mesh, the config and the plan inputs: computed once
// per resources (MoEBlockResources::plan) and shared by every call.
struct CombinePlan {
    MeshTopology topo;
    MeshPlan mesh;  // ports: peers and fabric link ids
    uint32_t L = 0;
    uint32_t snap_bits = 0;
    uint32_t snap_hi_bits = 0;  // MoEBlockConfig::combine_snap_split: C_hi's fp32 bits (0: one fp32 sum a token)
    uint32_t acc_sums = 1;      // fp32 sums a token in CB_ACC (2 with the split: lo, then hi)
    uint32_t S = 0, W = 0;
    RowSlot pc;
    StateLayout st;
    uint32_t A = 0, MAXT = 0, PSL = 0, D = 0, QP = 0, QL = 0;
    bool fused = false;    // acc_on_ports
    bool perchip = false;  // per-chip accumulator roles (ports apart): the senders' DRAM LUT
    bool shared = false;   // the shared landing pool (CombineLanding::Shared)
    // CombineLanding::FlatCores: accumulator o's ring slot i is pool slot o x PSL + i of the flat cells' free arena,
    // pool_args its description (kernels/dataflow/combine_pool.hpp; the same cells on every chip), pool_slots its slots
    bool flat_pool = false;
    uint32_t pool_slots = 0;
    uint32_t row_depth = 2;          // the accumulators' row / meta / weight CBs (rows): kPoolRows with flat_pool
    uint32_t tab_acc = kTabCbBytes;  // the accumulators' CB_TAB bytes (acc_tab_bytes)
    std::vector<uint32_t> pool_args;
    CreditPolicy cred;
    CombineAutoChoice choice;

    // Arena layout (arena-relative offsets; sparse_combine_rf_prepare's placement).
    uint32_t arena_bytes = 0;  // the combine's bytes a core; the y writers' words follow
    uint32_t inj_off = 0;      // the port's injection ring (ports apart: RINGS; acc_on_ports: after the local ring)
    uint32_t lut_off = 0, lut_bytes = 0;
    uint32_t tab_off_port = 0, tab_off_acc = 0;  // CB_TAB on the ports / accumulators
    uint32_t row_off = 0, meta_off = 0, acc_off = 0, out_off = 0, w_off = 0;
    uint32_t loc_off = 0;     // the accumulators' local ring
    uint32_t fill_chunk = 0;  // MOE_FILL=piggyback: a fill piece's bytes (expert_slot_copy's chunk: the fabric payload)

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
    // cached_slots (P.63): each patch's runtime-args record in this program, filled on the first such hit (see
    // FlatProgramHandles::slots). Empty: not filled yet.
    mutable std::vector<tt::tt_metal::RuntimeArgsData*> slots;
    // The y writers' report (flat_program.hpp FlatCombineReport): the collector (virtual), its counts (arena offset),
    // and the L1 hand-off args (y_row_major.hpp SE_Y_L1): [nd, slot stride, table, words, writer index, gate x | y <<
    // 16, bar, words bytes to zero; per destination: noc x, noc y, slot base, slots, landed counter array].
    tt::tt_metal::CoreCoord collector;
    uint32_t counts_offset = 0;
    std::vector<CombineArg> l1_args;
    // MOE_FILL=piggyback: per port of this chip (port id order) the sender's kernel, core and first fill arg index
    struct FillPort {
        tt::tt_metal::KernelHandle kernel;
        tt::tt_metal::CoreCoord core;
        uint32_t index;
        uint32_t dir, li;
    };
    std::vector<FillPort> fill_ports;
};

// The combine's kernels, CBs and runtime args on chip `chip` of `program` (the flat expert's go on its own cells).
CombineHandles add_combine_program(
    tt::tt_metal::Program& program,
    const CombinePlan& plan,
    ChipCoord chip,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineBuffers& buffers,
    const FlatHandoff& flat,
    bool fill = false,
    bool fill_exact = false);

// MOE_FILL=piggyback: the port senders' fill args of chip `chip` for this call (call: none = no job; done: the fill's
// done words' address). Written on every call (create and cache hit).
void set_combine_fill_args(
    tt::tt_metal::Program& program,
    const CombineHandles& h,
    const CombinePlan& plan,
    ChipCoord chip,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CombineFillCall* call,
    uint32_t done_addr,
    bool exact = false,
    uint32_t token = 0);

// MOE_FILL=piggyback: the arena offset of a port's fill table and staging slots (they end at the arena's end).
uint32_t combine_fill_offset(const CombinePlan& plan, bool exact = false);

// Re-applies the buffers' addresses (a program-cache hit): a source's args (and the arena's CBs) only when its address
// moved.
// cached_slots (P.63): through h.slots (the same values; no kernel lookup a patch).
void override_combine_addresses(
    tt::tt_metal::Program& program, const CombineHandles& h, const CombineBuffers& buffers, bool cached_slots = false);

// The writers' l1_args with the buffers' current addresses.
std::vector<uint32_t> resolve_l1_args(const CombineHandles& h, const CombineBuffers& buffers);

// The sender's DRAM header cache page bytes (N x NH routed headers + a 64 B key).
inline uint32_t hcache_page_bytes(uint32_t N) { return N * kNumHeaders * kHeaderStride + 64; }

}  // namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine
