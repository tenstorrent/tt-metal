// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math of the sparse combine (no device): the row's landing slot, the per-core state layout, the L1 sizing and the
// automatic configuration (combine_auto), the credit policy and the exchange's arena reserve. Checked by
// tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_routed_experts_combine_host.py.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"
#include "ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/dataflow/combine_formats.hpp"
#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

inline constexpr uint32_t kTile = 2048;         // bf16 tile bytes (rows are handled as flat tiles)
inline constexpr uint32_t kNumHeaders = 8;      // NH: rotating row headers per destination chip
inline constexpr uint32_t kHeaderStride = 128;  // HSZ
// the MISC block of the state (its words: kernels/dataflow/combine_formats.hpp)
inline constexpr uint32_t kMiscBytes = combine_formats::kMiscBytes;
inline constexpr uint32_t kMiscRtot = combine_formats::kMiscRowsConsumed;
inline constexpr uint32_t kMiscRdone = combine_formats::kMiscRowsReturned;
inline constexpr uint32_t kMiscTaken = combine_formats::kMiscTaken;
inline constexpr uint32_t kAccMax = moe_block_plan_words::kAccMax;  // A_MAX: accumulators at most (the plan's want[])
inline constexpr uint32_t kMaxChips = moe_block_plan_words::kMaxChips;  // N at most (want[][32], 32-bit chip masks)
// CB_TAB: the device plan's scratch reads (accumulators: the plan's words, the shared landing's slot owners / free list
// / grant queues, the flat expert cores' trailer probes - accumulator_reader.cpp)
inline constexpr uint32_t kTabCbBytes = 256 * 10 + 128;
inline constexpr uint32_t kWriterWordsRows = 192;       // the L1 hand-off writers' words: 64 B rows
// The kernels' args blocks (common/args_table.hpp runtime_args_to_dram_fixup), past the writers' words on every core of
// the arena: the RISCV_0 kernel's block in the first half, the RISCV_1 kernel's in the second (the largest block - a
// port-or-accumulator core's sender: its own args, the accumulator writer's block, the fabric connection, the barrier's
// port list - must fit a half; the converter fails loudly, naming the kernel and the bytes, when one does not)
inline constexpr uint32_t kArgsRows = 128;
inline constexpr uint32_t kArgsHalfBytes = kArgsRows * 64 / 2;
// the args region's arena offset: past the combine's bytes and the writers' words
inline uint32_t args_offset(uint32_t arena_bytes) { return arena_bytes + kWriterWordsRows * 64; }
// shared landing: slot grants outstanding a (source, accumulator) at most (4: 2x4 GLM-5.3-Flash 1280 experts + combine
// 560 / 694 / 1157 us at D 8 / 2 / 1 vs 584 / 688 / 1172 at 8, 591 / 699 / 1194 at 16)
inline constexpr uint32_t kGrants = 4;
// the y writers' L1 hand-off destinations at most (y_row_major.hpp combine_handoff::kMaxDestinations; flat_program.hpp
// kFlatYL1Args): a chip's ports and its accumulators together
inline constexpr uint32_t kHandoffDests = 32;
inline constexpr uint32_t kSharedPslMax = 254;  // shared landing in the accumulators' L1: slots an accumulator at most
inline constexpr uint32_t kSharedFlatPslMax = 1024;  // ... on the flat expert cores (slot ids 16-bit, owners bytes)

// Plan tensor word offsets (common/kernels/combine_plan_words.hpp, shared with the kernels).
inline constexpr uint32_t kPlanWords = moe_block_plan_words::kPlanWords;
inline constexpr uint32_t kPlanNinj = moe_block_plan_words::kPlanNinj, kPlanNret = moe_block_plan_words::kPlanNret,
                          kPlanNloc = moe_block_plan_words::kPlanNloc, kPlanWant = moe_block_plan_words::kPlanWant,
                          kPlanRows = moe_block_plan_words::kPlanRows, kPlanOver = moe_block_plan_words::kPlanOver;

inline constexpr uint32_t align64(uint64_t v) { return static_cast<uint32_t>((v + 63) / 64 * 64); }

// A W-wide bf16 row's landing slot: the row (one fabric packet), its 32 B trailer at row_bytes, 64 B aligned.
struct RowSlot {
    uint32_t row_bytes = 0;
    uint32_t NT = 0;    // bf16 tiles a row
    uint32_t slot = 0;  // slot stride
    bool operator==(const RowSlot&) const = default;
};
RowSlot row_slot(uint32_t W);

// _state: byte offsets of the per-core state block; RINGS is its end (the landing pool starts there). G: the shared
// landing's grant ring (0: rings, no grant words): GIDH (at a home port: per (source, accumulator) G grant ids) and
// GREC (at a source port: per home its snapshot - AP grant counts, then AP x G ids); the credit snapshots' staging
// (one per header rotation) grows to a grant snapshot's AP x (1 + G) words.
struct StateLayout {
    uint32_t CRED = 0, USED = 0, RET = 0, LAYR = 0, LAYO = 0, U0 = 0, CNT = 0, MISC = 0, HDRS = 0, RINGS = 0;
    uint32_t GIDH = 0, GREC = 0, G = 0;
    bool operator==(const StateLayout&) const = default;
};
StateLayout state_layout(uint32_t N, uint32_t A, uint32_t G = 0);

// _acc_cbs: an accumulator's CB bytes (CB_TAB, CB_ROW, CB_META, CB_ACC, CB_OUT, CB_W; each 64 B aligned).
uint32_t acc_cb_bytes(uint32_t S, uint32_t A, uint32_t NT, uint32_t rows = 2, uint32_t tab = kTabCbBytes);
// an accumulator's CB_TAB bytes: the shared landing's slot owners (1 B) and free list (2 B a slot) last
// (accumulator_reader.cpp: plan words, grant queues, trailer probes, zeros, then them from 1600 B)
inline uint32_t acc_tab_bytes(bool shared, uint32_t psl) {
    return shared ? std::max<uint32_t>(kTabCbBytes, align64(1600 + 4 + 3 * psl + 64)) : kTabCbBytes;
}
// the accumulator's row CB depth (rows, its meta and weight CBs too) with the rings on the flat expert cores: its row
// reads cross the NoC, kPoolRows of them in flight
inline constexpr uint32_t kPoolRows = 4;

// combine_l1_layout: what the combine puts in each core's arena budget (bytes). Pure.
struct CombineLayout {
    uint32_t slot = 0;
    uint32_t acc_fp32 = 0;
    uint32_t acc_cbs = 0;
    uint32_t rings = 0;  // the state block's bytes (RINGS)
    int32_t PSL = 0;     // landing pool slots per accumulator (negative: no room)
    int32_t D = 0;       // PSL / (N - 1), floored
    uint32_t QP = 0;     // a port's injection ring (slots)
    uint32_t QL = 0;     // an accumulator's local ring (slots)
    uint32_t acc_used = 0;
    uint32_t port_used = 0;
    bool has_PSL = false;    // PSL / D computed (the reference's dict carries them)
    bool has_rings = false;  // QP / QL / acc_used / port_used computed
    bool fits = true;
    std::string why;
};
CombineLayout combine_l1_layout(
    uint32_t N,
    uint32_t S,
    uint32_t W,
    uint32_t A,
    uint32_t arena_bytes,
    bool acc_on_ports = false,
    uint32_t inj_slots = 64,
    uint32_t depth = 8,
    uint32_t local_keep = 8,
    bool shared = false,
    uint32_t pool_slots = 0);

enum class CombineMode : uint8_t { PortsApart = 0, AccOnPorts = 1 };

// The landing pool at an accumulator: Rings - carved per call into one ring per source chip (a source's rows go to its
// ring's next slot, credits return consumed slots); Shared - one free list, the accumulator grants specific slots to
// the sources with rows pending (ids ride the credit snapshots, tag-validated), so a burst from one source uses slots
// another source's ring would hold empty (accumulators apart from or on the ports). FlatCores - rings as Rings,
// their slots in the flat expert cores' free L1 (each flat core's arena past the flat plan's bytes) instead of the
// accumulator's: the accumulator keeps only its fp32 sums and CBs there, it learns a row landed by reading the slot's
// trailer and takes the row with a NoC read (as it takes a local slot's), so deep rings cost the accumulators nothing.
// (pool_slots: the flat cores' slots a chip, every accumulator's rings carved from them; 0 - in the accumulators' L1)
// SharedFlatCores - Shared's slot grants, the slots in the flat cores' free L1 as FlatCores' (up to kSharedFlatPslMax
// slots an accumulator).
enum class CombineLanding : uint8_t { Rings = 0, Shared = 1, FlatCores = 2, SharedFlatCores = 3 };
inline bool landing_shared(CombineLanding l) {
    return l == CombineLanding::Shared || l == CombineLanding::SharedFlatCores;
}
inline bool landing_on_flat(CombineLanding l) {
    return l == CombineLanding::FlatCores || l == CombineLanding::SharedFlatCores;
}

// combine_auto: the combine's configuration from the cores the flat plan leaves (free_core_count, port_core_count of
// them the ports') and the arena budget. mode: that mode only.
struct CombineAutoChoice {
    bool acc_on_ports = false;
    uint32_t n_acc = 0;
    uint32_t inj_slots = 0;
    uint32_t depth = 0;
    uint32_t local_keep = 0;
    bool shared = false;  // CombineLanding::Shared
    CombineLayout layout;
};
// How combine_auto / the landing choice rank configurations: landing depth up to 3 slots a source, then keeping at
// least half the most accumulators of the candidates (a_ref), then the injection ring (to 64 slots), then deeper
// landing, more accumulators, the local ring. Galaxy GLM-5.3 (2026-11-05): wherever the depth-first ranking put the
// accumulators on the ports for one more slot a source (an 8-slot injection ring), them apart (64 slots) was faster
// at D >= 3 - 4x4 5120 1.43 / 1.34 vs 1.57 / 1.59 ms, 4x4 7680 on the flat expert cores 2.43 vs 2.93 - but apart at D 1
// slower (2.369 vs 2.105); on the 2x4 LoudBox apart leaves 3 accumulators of 9 (GLM-5.3 1280 666 vs 645 us).
std::tuple<uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t> combine_choice_key(
    const struct CombineAutoChoice& c, uint32_t a_ref);

struct CombineAuto {
    bool fits = false;
    std::string why;
    std::optional<CombineAutoChoice> best;
    std::optional<CombineAutoChoice> ports_apart;   // modes["ports apart"]
    std::optional<CombineAutoChoice> acc_on_ports;  // modes["acc_on_ports"]
};
CombineAuto combine_auto(
    uint32_t N,
    uint32_t S,
    uint32_t W,
    uint32_t free_core_count,
    uint32_t port_core_count,
    uint32_t arena_bytes,
    uint32_t depth = 8,
    uint32_t a_max = kAccMax,
    std::optional<CombineMode> mode = std::nullopt,
    CombineLanding landing = CombineLanding::Rings,
    uint32_t pool_slots = 0);

// The credit policy: a port returns a source's credits once it consumed CRED_THR of its rows; its idle credit
// snapshots are batched (CRED_IDLE rows, or CRED_AGE cycles old). N chips, D landing slots a source chip (average);
// batch: CombineConfig.credit_batch (rows a snapshot covers, 1 at once), else the planner's.
struct CreditPolicy {
    uint32_t cred_thr = 8;
    uint32_t cred_idle = 8;
    uint32_t cred_age = 8100;
    bool operator==(const CreditPolicy&) const = default;
};
CreditPolicy credit_policy(
    uint32_t N = 32, uint32_t D = 16, bool shared = false, std::optional<uint32_t> batch = std::nullopt);

// acc_on_ports_arena_reserve / exchange_arena_reserve: bytes a bank keeps below the combine's arena over a call (the
// static CBs of the programs that run while it is allocated, the y writers' words, slack), and the arena budget (bytes
// a core) from the largest free L1 block.
uint32_t acc_on_ports_arena_reserve(uint32_t flat_static_bytes);
uint32_t exchange_arena_reserve(uint32_t static_cb_bytes);
uint32_t exchange_arena_bytes(uint32_t largest_free_bytes, uint32_t reserve);

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
