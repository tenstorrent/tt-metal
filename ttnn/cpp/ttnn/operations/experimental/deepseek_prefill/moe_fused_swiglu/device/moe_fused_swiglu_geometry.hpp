// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry {

// ===========================================================================
// THE L1 BUDGET — what every parameter below costs, and where
// ===========================================================================
//
// Per-core CB bytes must fit `hal::get_max_worker_l1_unreserved_size() - L1_CB_RESERVE`
// (1 532 032 - 70 656 = 1 461 376 B on Blackhole). `cb_layout()` below is the table; the program
// factory sums it with the REAL scratch page sizes and refuses with both numbers rather than
// letting the allocator throw, so a no-fit is a readable message rather than a crash.
//
// `Blocking::l1_bytes()` is the same sum with the two DRAM scratch CBs assumed to be one 64 B page
// each. The constructor has to use it — it runs before the buffers are known — so the fallback
// ladder decides on a figure that can UNDERSTATE the real total by up to
// (idx_page - 64) + (counts_page - 64), typically ~960 B for a 256-expert counts vector. The
// factory's check is the exact one; treat l1_bytes() as the ladder's estimate, not the guard.
//
// TWO CALLER INPUTS also move L1, neither of them a knob:
//   `m_t_max` (= input_m_tiles, the caller's capacity in tile-rows) decides `max_m_blocks`, and a
//   single-M-block dispatch forces depth_x to 1 — so a SMALLER capacity can use LESS L1.
//   The activation LAYOUT picks CB_X_IN's page: 32 x_stick bytes row-major against one bfp8 tile
//   tiled, which is why the two formats differ by tens of KB at the same shape.
//
// THE FOUR DERIVED WIDTHS do the actual scaling. None is a knob; each falls out of the shape and
// the grid, and every CB is a product of them:
//
//   kr_pad    = ceil(emb_t / kgroups)      K tiles per grid ROW    -> x and gate/up weights
//   hn_pad    ~ ceil(hid_t / hgroups)      hidden tiles per COLUMN -> gate/up weights, h, accs
//   ec_max    = max split(emb_t, cores)    down output width       -> W_down, out
//   wd_ec_max = wd_mgroups ? ec_group_max : ec_max                 -> W_down, out
//
// So L1 grows with `emb/kgroups` and `hidden/hgroups`: a WIDER grid shrinks hn_pad, a TALLER grid
// shrinks kr_pad. `hn_pad` is chosen by choose_hn_pad(), which is NOT L1-aware — it optimises the
// gate/up chunk count, and an awkward hn_pad inflates `slice_pages` (an lcm) disproportionately.
//
// THE CB TABLE (pages x page size; see cb_layout() for the code):
//
//   CB_X_IN          rm ? 32 : 1              x_stick   staging sticks; x_stick = kr_pad*32*elem
//   CB_X_TILES       depth_x * M_BLOCK*kr_pad bfp8      tilized x, the resident in0 for gate/up
//   CB_W_GATE/_UP    depth_w * kr_pad*hn_pad  w_tile    resident at W_RESIDENT (depth_w == 1)
//   CB_W_DOWN        depth_wd * hn_pad*wd_ec_max w_tile resident at WD_RESIDENT (depth_wd==hgroups)
//   CB_H             depth_h * h_fast         bfp8      h_fast = wd_mrow ? hid_t : M_BLOCK*hn_pad
//   CB_GATE_ACC/_UP  M_BLOCK * hn_pad         bfp8      one whole gate/up output block, each
//   CB_H_LOCAL       max(M_BLOCK*hn_pad, h_fast) bfp8   the column's assembled h
//   CB_GATHER_*      gather_pages             bfp8      reduce-scatter landing, each
//   CB_SLICE_*       slice_pages              bf16      reduce-scatter operands, each
//   CB_H_SLICE       slice_pages              bfp8
//   CB_GATE_SILU     slice_pages              bf16
//   CB_OUT_TILES     DEPTH_OUT * out_block    out_tile  out_block = M_BLOCK*ec_max (or grouped)
//   CB_OUT_INTERM    (wd_mrow?M_BLOCK/2:M_BLOCK)*ec_max bf16  K-blocking accumulator
//   CB_IDX/_COUNTS   1                        page      DRAM scratch, aligned page each
//   CB_X_STAGE + the two mailbox views        64 B      ONE 64 B allocation, three FIFO views
//
// EIGHT CBs SCALE WITH `M_BLOCK * hn_pad` unconditionally: both accs directly, and the two GATHERs
// plus SLICE_GATE/SLICE_UP/H_SLICE/GATE_SILU through gather_pages and slice_pages, which are both
// derived from `m * hn_pad` over m <= M_BLOCK. CB_H and CB_H_LOCAL join them only when
// `wd_mrow_rounds` is OFF; with it on they follow `hid_t` instead and M_BLOCK does not move them.
// That product is still the single biggest L1 term. NOTE the ladder below does NOT touch M_BLOCK:
// halving it is a source change, and the last resort only after every ladder step has been spent.
//
// ALIASING (cb_allocations) overlays CBs whose lifetimes do not overlap onto one allocation sized
// to the lcm of their page counts. Three groups: the mailbox trio (always), the phase group
// (GATHER_GATE/H_SLICE/OUT_TILES, only when the output is bfp8 so the page sizes agree), and the
// bf16 pair (GATE_SILU/OUT_INTERM). Each is applied only when the lcm beats the sum.
//
// TUNING OVERRIDES: every knob marked [env] below can be overridden at process start; see env_u32 /
// env_bool in the .cpp. Defaults reproduce the shipped geometry exactly.
// ===========================================================================

inline constexpr uint32_t TILE = 32;

//: Token tile-rows per outer M-block. Not env-tunable. THE master L1 scale factor — eight CBs are
//: sized `M_BLOCK * hn_pad` (see the table above). Must be a power of two so a power-of-two `m_eff` divides it and
//: every CB reserve stays block-aligned (see m_tiles_eff in the kernels' common.hpp).
inline constexpr uint32_t M_BLOCK = 8;

//: DEST register tiles available to one matmul. Not an L1 term, but it CAPS `ec_max`, which is an
//: L1 term: the constructor refuses a grid whose down-output width exceeds it.
inline constexpr uint32_t DEST_LIMIT = 8;

//: Matmul sub-block heights. `OUT_SUBBLOCK_H_GU` also sets `m_eff_min`, the smallest legal runtime
//: M-block, which is the lower bound on every M-scaled CB. No direct L1 term of their own.
inline constexpr uint32_t OUT_SUBBLOCK_H_GU = 1;
inline constexpr uint32_t OUT_SUBBLOCK_H_DN_MAX = 4;

//: SFPU eltwise batch size. Register-level only; no L1 term.
inline constexpr uint32_t ELTWISE_BLK = 8;

//: CB DEPTHS — the producer/consumer pipeline slots. Each multiplies its CB directly.
//: DEPTH_W is bypassed whenever W_RESIDENT (depth_w collapses to 1).
inline constexpr uint32_t DEPTH_W = 2;
//: [env MOE_DEPTH_X] x slots. 2 buys the cross-M-block prefetch; 1 costs ~0.2% and frees
//: `M_BLOCK * kr_pad` bfp8 tiles, which is often the cheapest way to buy an h slot back.
inline constexpr uint32_t DEPTH_X = 2;
//: [env MOE_DEPTH_H, 2..MAX_DEPTH_H] h slots. The largest single depth term: `depth_h * h_fast`
//: bfp8 tiles.
inline constexpr uint32_t DEPTH_H = 3;
//: HARD CAP, not a preference: the per-slot VALID cells are semaphores SEM_H_RDY_BASE + slot, and
//: SEM_H_FREE sits immediately above them. A deeper h pipeline would alias two protocols onto one
//: semaphore. Raising it means moving the semaphore map first.
inline constexpr uint32_t MAX_DEPTH_H = 3;
//: Output slots. Double-buffers CB_OUT_TILES so the writer drains block b while compute fills b+1.
inline constexpr uint32_t DEPTH_OUT = 2;

//: Rows of x sticks staged at once. Multiplies CB_X_IN, which only exists for row-major input.
inline constexpr uint32_t XSTICK_ROWS = 1;

//: [env MOE_WD_AHEAD] W_down K-blocks kept in flight. Bounds `depth_wd` from below
//: (depth_wd >= wd_ahead + 2), so raising it raises the CB_W_DOWN floor.
inline constexpr uint32_t WD_AHEAD = 1;

//: [env MOE_WD_MROW] Whole-h-row phase-2 schedule. Requires kgroups == M_BLOCK. Sets
//: `h_fast = hid_t` instead of `M_BLOCK * hn_pad`, so it re-sizes CB_H and CB_H_LOCAL.
inline constexpr bool WD_MROW_ROUNDS = true;

//: [env MOE_WD_MGROUPS] Paired-row grouped broadcast: two concurrent half-grid h schedules, so a
//: sender waits on num_cores/2 acks instead of num_cores. Costs L1 by widening `wd_ec_max` from
//: `ec_max` to `ec_group_max`, which inflates CB_W_DOWN. CB_OUT_TILES is unaffected — its grouped
//: term `mgroup_rows * ec_group_max` never exceeds the ordinary `M_BLOCK * ec_max`.
inline constexpr bool WD_MGROUPS = false;
//: [env MOE_WD_MGROUP_MIN_BLOCKS] Runtime gate: grouped mode engages only at m_blocks >= this.
//: Not an L1 term — the CBs are provisioned for grouped mode whenever WD_MGROUPS is on.
inline constexpr uint32_t WD_MGROUP_MIN_BLOCKS = 4;

//: Cross-M-block weight residency. For GATE/UP it SAVES L1 — depth_w collapses from 2 to 1.
//: For W_DOWN it COSTS L1: a resident shard is depth_wd == hgroups slots, where streaming needs
//: only wd_ahead + 2. That is why dropping WD_RESIDENT is the ladder's biggest single reclaim.
inline constexpr bool W_RESIDENT = true;
inline constexpr bool WD_RESIDENT = true;

//: [env MOE_GU_CHUNKS] Preferred gate/up N-chunk count. Not an L1 term itself, but it selects
//: `hn_pad` via choose_hn_pad(), and hn_pad is in almost every CB. A chunk width of 1 tile is
//: legal and very slow — measured +35% at 11x8 emb 3584.
inline constexpr uint32_t GU_CHUNKS = 3;

//: Transport knobs — scheduling only, no L1 term.
inline constexpr bool XPRIO = true;
inline constexpr uint32_t H_ROUND_NOC1_MASK = 0;
inline constexpr bool SCATTER_ONE_SIGNAL = true;
inline constexpr bool H_MCAST_POSTED = true;

//: h slots prefetched ahead of the round consuming them. Clamped to depth_h - 1, so it never
//: enlarges CB_H on its own.
inline constexpr uint32_t HACK_AHEAD = 2;

//: [env MOE_WD_SPLIT] W_down DRAM read split across the two NoCs. Requires a fully resident
//: W_down (depth_wd == hgroups). No L1 term; it needs one NoC transaction id per column.
inline constexpr uint32_t WD_SPLIT = 3;

inline constexpr uint32_t MAILBOX_MAGIC = 0xC0FFEE01;
inline constexpr uint32_t MAILBOX_WORDS = 16;

//: L1 held back from the CB budget for kernel stack, args and the profiler ring. Subtracted from
//: the worker unreserved size to give `l1_budget`; raising it shrinks every shape's headroom.
inline constexpr uint32_t L1_CB_RESERVE = 70656;

inline constexpr uint32_t NOC_MAX_TRANSACTION_ID = 15;

inline constexpr uint32_t CB_X_IN = 0;
inline constexpr uint32_t CB_X_TILES = 1;
inline constexpr uint32_t CB_X_STAGE = 2;
inline constexpr uint32_t CB_W_GATE = 3;
inline constexpr uint32_t CB_W_UP = 4;
inline constexpr uint32_t CB_W_DOWN = 5;
inline constexpr uint32_t CB_H = 6;
inline constexpr uint32_t CB_IDX_SCRATCH = 7;
inline constexpr uint32_t CB_COUNTS_SCRATCH = 8;
inline constexpr uint32_t CB_GATHER_GATE = 9;
inline constexpr uint32_t CB_GATHER_UP = 10;
inline constexpr uint32_t CB_SLICE_GATE = 11;
inline constexpr uint32_t CB_SLICE_UP = 12;
inline constexpr uint32_t CB_H_SLICE = 13;
inline constexpr uint32_t CB_OUT_TILES = 14;
inline constexpr uint32_t CB_GATE_ACC = 15;
inline constexpr uint32_t CB_UP_ACC = 16;
inline constexpr uint32_t CB_GATE_SILU = 17;
inline constexpr uint32_t CB_H_LOCAL = 18;
inline constexpr uint32_t CB_OUT_INTERM = 19;
// Two independent FIFO views over CB_X_STAGE's 64-byte backing. Reader pushes
// both once after publishing the runtime-count mailbox; compute and writer
// consume one view each. CB_X_STAGE itself remains exclusively the per-row
// compute->reader tilization completion channel. No extra L1 allocation.
inline constexpr uint32_t CB_MAILBOX_WRITER = 20;
inline constexpr uint32_t CB_MAILBOX_COMPUTE = 21;

inline constexpr uint32_t SEM_X_BASE = 0;
inline constexpr uint32_t SEM_H_BASE = 2;
inline constexpr uint32_t SEM_GO = 4;
inline constexpr uint32_t SEM_DATA = 5;
inline constexpr uint32_t SEM_HSLICE = 6;
inline constexpr uint32_t SEM_XSTAGED = 7;
inline constexpr uint32_t SEM_H_RDY_BASE = 8;
inline constexpr uint32_t SEM_H_FREE = 11;
inline constexpr uint32_t SEM_WDSPLIT = 12;
inline constexpr uint32_t SEM_PHASE_FREE = 13;
inline constexpr uint32_t SEM_HROW_FREE = 14;
inline constexpr uint32_t SEM_COUNT = 15;
inline constexpr uint32_t NUM_DEVICE_SEMAPHORES = 16;

enum class FormatKey : uint8_t { Bfp8, Bf16, Weight, Out, U32, XIn };

struct CbView {
    uint32_t index;
    uint32_t pages;
    uint32_t page_size;
    FormatKey format;
};

struct CbAllocation {
    uint32_t total_size;
    std::vector<CbView> views;
};

struct ScatterPlan {
    uint32_t slice_pages = 0;
    uint32_t gather_pages = 0;
    std::vector<uint32_t> sizes;
};

class Blocking {
public:
    Blocking(
        uint32_t hgroups,
        uint32_t kgroups,
        uint32_t emb,
        uint32_t hidden,
        uint32_t m_t_max,
        uint32_t w_tile,
        uint32_t bfp8_tile,
        uint32_t bf16_tile,
        uint32_t x_stick,
        uint32_t l1_budget,
        uint32_t out_tile,
        bool enable_phase_alias,
        bool x_is_rm);

    std::vector<CbView> cb_layout(bool x_is_rm, uint32_t out_tile, uint32_t idx_page, uint32_t counts_page) const;
    std::vector<CbAllocation> cb_allocations(
        bool x_is_rm, uint32_t out_tile, uint32_t idx_page, uint32_t counts_page, bool enable_phase_alias) const;
    uint64_t l1_bytes(bool x_is_rm, uint32_t out_tile, bool enable_phase_alias) const;
    bool phase_cb_alias(uint32_t out_tile) const;
    uint32_t phase_cb_alias_pages(uint32_t out_tile) const;
    std::string describe() const;

    uint32_t hgroups;
    uint32_t kgroups;
    uint32_t num_cores;
    uint32_t emb;
    uint32_t hidden;
    uint32_t emb_t;
    uint32_t hid_t;
    uint32_t m_t_max;
    uint32_t m_eff_min;
    std::vector<uint32_t> kr_sizes;
    std::vector<uint32_t> kr_starts;
    uint32_t kr_pad;
    uint32_t hn_pad;
    uint32_t gu_chunks;
    uint32_t gu_chunks_target;
    uint32_t gu_chunk_w;
    uint32_t hn_block;
    uint32_t gu_in1_subblocks;
    bool balanced_hn;
    std::vector<uint32_t> hn_sizes;
    std::vector<uint32_t> hn_starts;
    bool wd_mrow_rounds;
    std::vector<uint32_t> ec_sizes;
    std::vector<uint32_t> ec_starts;
    uint32_t ec_max;
    uint32_t mgroup_rows;
    uint32_t mgroup_cores;
    std::vector<uint32_t> ec_group_sizes;
    std::vector<uint32_t> ec_group_starts;
    uint32_t ec_group_max;
    bool wd_mgroups;
    uint32_t wd_mgroup_min_blocks;
    uint32_t wd_ec_max;
    uint32_t out_subblock_h_dn;
    uint32_t gather_pages;
    uint32_t slice_pages;
    uint32_t max_m_blocks;
    uint32_t depth_x;
    uint32_t depth_w;
    uint32_t wd_ahead;
    uint32_t depth_h;
    uint32_t hack_ahead;
    uint32_t w_tile;
    uint32_t bfp8_tile;
    uint32_t bf16_tile;
    uint32_t x_stick;
    uint32_t out_tile;
    bool enable_phase_alias;
    bool x_is_rm;
    uint32_t l1_budget;
    bool wd_resident;
    uint32_t depth_wd;
    bool wd_packed;
    uint32_t wd_split;

private:
    struct HnChoice {
        uint32_t hn_pad;
        uint32_t chunks;
        ScatterPlan plan;
        bool balanced;
    };

    HnChoice choose_hn_pad() const;
    bool depth_wd_legal(uint32_t depth) const;
    uint32_t min_depth_wd() const;
    uint32_t next_smaller_depth_wd(uint32_t depth) const;
};

uint32_t nd_shard_n_tiles(const Tensor& tensor);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry
