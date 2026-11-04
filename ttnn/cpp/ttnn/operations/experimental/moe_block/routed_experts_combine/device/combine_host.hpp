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

#include "ttnn/operations/experimental/moe_block/common/mesh_routing.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

inline constexpr uint32_t kTile = 2048;                 // bf16 tile bytes (rows are handled as flat tiles)
inline constexpr uint32_t kNumHeaders = 8;              // NH: rotating row headers per destination chip
inline constexpr uint32_t kHeaderStride = 128;          // HSZ
inline constexpr uint32_t kMiscBytes = 2048;            // MISC
inline constexpr uint32_t kMiscRtot = 256;              // MISC words: rows consumed per source chip
inline constexpr uint32_t kMiscRdone = 288;             // MISC words: rows covered by the credit snapshots sent
inline constexpr uint32_t kMiscTaken = 320;             // MISC words: an accumulator's taken[] per ring
inline constexpr uint32_t kAccMax = 16;                 // A_MAX: accumulators at most (state layout, device planner)
inline constexpr uint32_t kMaxChips = 32;               // N at most (the plan's want[16][32], 32-bit masks)
inline constexpr uint32_t kTabCbBytes = 256 * 8 + 128;  // CB_TAB: the device plan's scratch reads
inline constexpr uint32_t kWriterWordsRows = 192;       // the L1 hand-off writers' words: 64 B rows

// Plan tensor word offsets (combine_plan: P_NINJ, P_NRET, P_NLOC, P_WANT, P_ROWS, P_OVER).
inline constexpr uint32_t kPlanWords = 1024;
inline constexpr uint32_t kPlanNinj = 0, kPlanNret = 8, kPlanNloc = 16, kPlanWant = 32, kPlanRows = 600,
                          kPlanOver = 601;

inline constexpr uint32_t align64(uint64_t v) { return static_cast<uint32_t>((v + 63) / 64 * 64); }

// A W-wide bf16 row's landing slot: the row (one fabric packet), its 32 B trailer at row_bytes, 64 B aligned.
struct RowSlot {
    uint32_t row_bytes = 0;
    uint32_t NT = 0;    // bf16 tiles a row
    uint32_t slot = 0;  // slot stride
    bool operator==(const RowSlot&) const = default;
};
RowSlot row_slot(uint32_t W);

// _state: byte offsets of the per-core state block; RINGS is its end (the landing pool starts there).
struct StateLayout {
    uint32_t CRED = 0, USED = 0, RET = 0, LAYR = 0, LAYO = 0, U0 = 0, CNT = 0, MISC = 0, HDRS = 0, RINGS = 0;
    bool operator==(const StateLayout&) const = default;
};
StateLayout state_layout(uint32_t N, uint32_t A);

// _acc_cbs: an accumulator's CB bytes (CB_TAB, CB_ROW, CB_META, CB_ACC, CB_OUT, CB_W; each 64 B aligned).
uint32_t acc_cb_bytes(uint32_t S, uint32_t A, uint32_t NT);

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
    uint32_t local_keep = 8);

enum class CombineMode : uint8_t { PortsApart = 0, AccOnPorts = 1 };

// combine_auto: the combine's configuration from the cells the flat plan leaves (n_cells, port_cells of them the
// ports') and the arena budget. mode: that mode only.
struct CombineAutoChoice {
    bool acc_on_ports = false;
    uint32_t n_acc = 0;
    uint32_t inj_slots = 0;
    uint32_t depth = 0;
    uint32_t local_keep = 0;
    CombineLayout layout;
};
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
    uint32_t n_cells,
    uint32_t port_cells,
    uint32_t arena_bytes,
    uint32_t depth = 8,
    uint32_t a_max = kAccMax,
    std::optional<CombineMode> mode = std::nullopt);

// The credit policy: a port returns a source's credits once it consumed CRED_THR of its rows; idle snapshots go out
// at once on small meshes, batched (CRED_IDLE rows, or CRED_AGE cycles old) on meshes of 16 chips or more.
struct CreditPolicy {
    uint32_t cred_thr = 8;
    uint32_t cred_idle = 1;
    uint32_t cred_age = 0;
    bool operator==(const CreditPolicy&) const = default;
};
CreditPolicy credit_policy(uint32_t N);

// fused_arena_reserve / exchange_arena_reserve: bytes a bank keeps below the combine's arena over a call (the static
// CBs of the programs that run while it is allocated, the y writers' words, slack), and the arena budget (bytes a
// core) from the largest free L1 block.
uint32_t fused_arena_reserve(uint32_t flat_static_bytes);
uint32_t exchange_arena_reserve(uint32_t static_cb_bytes);
uint32_t exchange_arena_bytes(uint32_t largest_free_bytes, uint32_t reserve);

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
