// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host math of the sparse combine (no device): row pieces, the per-core state layout, the L1 sizing and the automatic
// configuration (combine_auto), the credit policy, the exchange's arena reserve, and the host model of the device
// planner (plan_reference). Faithful ports of ttnn/ttnn/bringup/combine_rf/{sparse_combine_rf,combine_plan}.py and
// models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py on mstaletovic/moe-trim; compared against them by
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
inline constexpr uint32_t kPieceMax = 7168;             // row bytes per fabric packet at most (default piece_max)
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
inline constexpr uint32_t kPlannerCores = 32;           // the exchange planner's cores (k3 PLANNER_CORES)

// Plan tensor word offsets (combine_plan: P_NINJ, P_NRET, P_NLOC, P_WANT, P_ROWS, P_OVER).
inline constexpr uint32_t kPlanWords = 1024;
inline constexpr uint32_t kPlanNinj = 0, kPlanNret = 8, kPlanNloc = 16, kPlanWant = 32, kPlanRows = 600,
                          kPlanOver = 601;

inline constexpr uint32_t align64(uint64_t v) { return static_cast<uint32_t>((v + 63) / 64 * 64); }

// _pieces: W-wide bf16 rows as NP equal pieces (64 B multiples), each with a 32 B trailer, PS apart.
struct RowPieces {
    uint32_t row_bytes = 0;
    uint32_t NT = 0;    // bf16 tiles a row
    uint32_t NP = 0;    // pieces a row
    uint32_t PD = 0;    // bytes a piece
    uint32_t PS = 0;    // piece stride
    uint32_t slot = 0;  // NP x PS
    bool operator==(const RowPieces&) const = default;
};
RowPieces row_pieces(uint32_t W, std::optional<uint32_t> piece_max = std::nullopt);

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
    std::optional<uint32_t> piece_max = std::nullopt,
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
    std::optional<uint32_t> piece_max = std::nullopt,
    uint32_t a_max = kAccMax,
    std::optional<CombineMode> mode = std::nullopt);

// The credit policy (sparse_combine_rf_prepare): idle snapshots batched on large meshes, SPC_CREDIDLE / SPC_CREDAGE
// override, SPC_CREDAUTO's end-only policy when every pool holds its call. max_rows_per_token: the most rows any
// (home, token) takes - the exchange passes no rows_by_chip, so 0 (the bound always holds).
struct CreditPolicy {
    uint32_t cred_thr = 8;
    uint32_t cred_idle = 1;
    uint32_t cred_age = 0;
    bool end_only = false;  // cred_policy "end"
    bool operator==(const CreditPolicy&) const = default;
};
CreditPolicy credit_policy(
    uint32_t N, uint32_t S, uint32_t A, uint32_t PSL, const CombineTuning& tuning, uint32_t max_rows_per_token = 0);

// fused_arena_reserve / exchange_arena_reserve (k3): bytes a bank keeps below the exchange's arena over a call, and
// the arena budget (bytes a core) the reference's exchange gets from the largest free L1 block.
uint32_t fused_arena_reserve(uint32_t flat_static_bytes);
// dispatch_static_bytes: sparse_all_gather_2d.static_cb_bytes(N, S, latent, group); manifest_bytes: its per-call
// manifest shard (64 B aligned).
uint32_t exchange_arena_reserve(
    uint32_t flat_static_bytes, uint32_t dispatch_static_bytes, uint32_t manifest_bytes, bool dispatch_in_program);
uint32_t exchange_arena_bytes(uint32_t largest_free_bytes, uint32_t reserve);
// sparse_all_gather_2d.static_cb_bytes: the dispatch's static CB bytes on its port cores.
uint32_t dispatch_static_cb_bytes(
    uint32_t N,
    uint32_t S,
    uint32_t row_width,
    uint32_t group = 8,
    uint32_t manifest_batch = 16,
    uint32_t dtype_bytes = 2);

// plan_reference (combine_plan): the device planner's per-call tables for chip `me`, host model. ids / wbits: [N][S][K]
// flattened (global expert ids, bf16 bits of the gate weights). lut: the chip's port LUT (direction * 2 + link).
struct PlanTables {
    std::vector<uint32_t> counts;       // [NG]
    std::vector<uint32_t> regions;      // [NG]
    std::vector<uint32_t> token_index;  // [rows_cap]
    std::vector<uint32_t> table;        // [2 rows_cap]
    std::vector<uint32_t> plan;         // [1024]
};
PlanTables plan_reference(
    const std::vector<uint16_t>& ids,
    const std::vector<uint16_t>& wbits,
    uint32_t me,
    uint32_t R,
    uint32_t C,
    uint32_t S,
    uint32_t K,
    uint32_t EPC,
    uint32_t NG,
    uint32_t A,
    uint32_t L,
    uint32_t P,
    const std::array<uint8_t, 8>& lut,
    uint32_t rows_cap,
    bool wrap_x,
    bool wrap_y,
    bool token_major,
    FabricTieRule tie);

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
