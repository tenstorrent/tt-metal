// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/common_globals.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/debug/assert.h"

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file eltwise_chain.hpp
 * @brief Core types for the redesigned eltwise helper family.
 *
 * Lives in `compute_kernel_lib::eltwise` to coexist with the legacy
 * `sfpu_helpers.hpp` / `binary_op_helpers.hpp` while migration is in flight.
 *
 * Provides:
 *   - Dst slot enum (D0..D15, hw ceiling)
 *   - Approx / Legacy bool-replacement enums
 *   - BroadcastDim enum (caller passes explicitly; never inferred)
 *   - CRTP bases UnaryOp / BinaryOp / TernaryOp (provide max_dst, exec, apply)
 *   - CopyTileTag + chain shape traits (clashes_with_fpu, is_upfront, …)
 *   - CopyTile element with 6-corner CopyTilePolicy (incl. CumulativeWait)
 *   - FillScalar / FillConst / CopyDest core elements
 *   - EltwiseChain<Ops...> + eltwise_chain(...) factory (no compaction)
 *   - eltwise_pipeline(...) — per-tile loop with FPU-clash-aware reinit
 *
 * Design rationale lives in
 *   ttnn/cpp/ttnn/kernel_lib/agents/eltwise_helper_lessons.md
 *   eltwise_helper_proposal.md
 *
 * ── DEST capacity ────────────────────────────────────────────────────────────
 *
 * Slot `D8..D15` are valid only in fp16 + SyncFull mode (16 tiles). The CRTP
 * bases static_assert against the absolute hw ceiling (16). The runtime
 * pipeline ASSERTs against `DEST_AUTO_LIMIT` (4 / 8 / 16 depending on sync +
 * accum mode) so a chain that uses D14 in fp32+halfsync fails at the acquire
 * site instead of silently wrapping.
 *
 * ── Examples ─────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
 *   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"     // for Exp
 *   using namespace compute_kernel_lib::eltwise;
 *
 *   // 1. Single op, streaming.
 *   auto chain = eltwise_chain(CopyTile<cb_in, Dst::D0>{}, Exp<>{});
 *   eltwise_pipeline(chain, cb_out, num_tiles);
 *
 *   // 2. Fan-out — y = x * exp(x).
 *   //    First CopyTile waits but does not pop; second skips wait, pops.
 *   //    Two physical copy_tile calls — N CopyTiles per fan-out (lessons §3.5).
 *   auto chain = eltwise_chain(
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},
 *       CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},
 *       Exp<Dst::D0>{},
 *       SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});                 // from eltwise_binary.hpp
 *   eltwise_pipeline(chain, cb_out, num_tiles);
 *
 *   // 3. Cumulative wait — producer-aligned read of all-prior-tiles.
 *   //    Caller pops the CB once at end-of-block (the helper does NOT pop).
 *   auto chain = eltwise_chain(
 *       CopyTile<cb_in, Dst::D0, CopyTilePolicy::CumulativeWait>{},
 *       Exp<>{});
 *   eltwise_pipeline(chain, cb_out, num_tiles);
 *   cb_pop_front(cb_in, num_tiles);
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

// =============================================================================
// 1. DEST slot enum + capacity helpers
// =============================================================================

/**
 * @brief Self-documenting DEST register slot indices. Hardware ceiling = 16.
 *
 * Active slot range depends on mode (lessons §1.3):
 *   SyncHalf + fp32 → D0..D3
 *   SyncHalf + fp16 → D0..D7
 *   SyncFull + fp32 → D0..D7
 *   SyncFull + fp16 → D0..D15
 */
enum class Dst : uint32_t {
    D0 = 0,
    D1 = 1,
    D2 = 2,
    D3 = 3,
    D4 = 4,
    D5 = 5,
    D6 = 6,
    D7 = 7,
    D8 = 8,
    D9 = 9,
    D10 = 10,
    D11 = 11,
    D12 = 12,
    D13 = 13,
    D14 = 14,
    D15 = 15,
};

constexpr uint32_t DST_HW_CEILING = 16;

// =============================================================================
// 2. Bool-replacement enums (named so they read at the call site)
// =============================================================================

/// Op approximation knob. Slow but exact = default. Opt into Fast.
enum class Approx : bool { Exact = false, Fast = true };

/// Legacy LLK overload selection (used by recip / rsqrt). Off = new path.
enum class Legacy : bool { Off = false, On = true };

// =============================================================================
// 3. Broadcast dimension (used by binary_op; declared here so the chain
//    pipeline can refer to it without depending on eltwise_binary.hpp).
// =============================================================================

/**
 * @brief Shape of operand B and how it broadcasts to match A.
 *
 * | BroadcastDim | B shape  | B tile count | Reduce that produces this output |
 * |--------------|----------|--------------|----------------------------------|
 * | NONE         | [Ht,Wt]  | Ht*Wt        | —                                |
 * | ROW          | [1,Wt]   | Wt           | REDUCE_COL                       |
 * | COL          | [Ht,1]   | Ht           | REDUCE_ROW                       |
 * | SCALAR       | [1,1]    | 1            | REDUCE_SCALAR                    |
 *
 * The helper does NOT infer broadcast from operand shape — the caller passes
 * the dim explicitly (lessons §10).
 */
enum class BroadcastDim { NONE, ROW, COL, SCALAR };

// =============================================================================
// 4. CRTP bases — provide max_dst, static_assert, exec(), apply()
//    Derived ops define ONLY init() and call(...).
// =============================================================================

template <typename Derived, Dst Slot>
struct UnaryOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static_assert(dst_idx < DST_HW_CEILING, "DEST slot exceeds hw ceiling (16)");

    // Default chain-shape traits — derived ops override as needed.
    static constexpr bool is_upfront = false;
    static constexpr bool is_cumulative = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t cb = 0;

    ALWI void exec(uint32_t offset = 0) const { static_cast<const Derived*>(this)->call(dst_idx + offset); }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

template <typename Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static constexpr uint32_t max_dst = (in0 > in1 ? (in0 > out ? in0 : out) : (in1 > out ? in1 : out));
    static_assert(in0 < DST_HW_CEILING, "DEST slot In0 exceeds hw ceiling (16)");
    static_assert(in1 < DST_HW_CEILING, "DEST slot In1 exceeds hw ceiling (16)");
    static_assert(out < DST_HW_CEILING, "DEST slot Out exceeds hw ceiling (16)");

    static constexpr bool is_upfront = false;
    static constexpr bool is_cumulative = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t cb = 0;

    ALWI void exec(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->call(in0 + offset, in1 + offset, out + offset);
    }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);

    static constexpr uint32_t _m01 = (in0 > in1 ? in0 : in1);
    static constexpr uint32_t _m23 = (in2 > out ? in2 : out);
    static constexpr uint32_t max_dst = (_m01 > _m23 ? _m01 : _m23);

    static_assert(in0 < DST_HW_CEILING, "TernaryOp In0 exceeds hw ceiling (16)");
    static_assert(in1 < DST_HW_CEILING, "TernaryOp In1 exceeds hw ceiling (16)");
    static_assert(in2 < DST_HW_CEILING, "TernaryOp In2 exceeds hw ceiling (16)");
    static_assert(out < DST_HW_CEILING, "TernaryOp Out exceeds hw ceiling (16)");

    static constexpr bool is_upfront = false;
    static constexpr bool is_cumulative = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t cb = 0;

    ALWI void exec(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->call(in0 + offset, in1 + offset, in2 + offset, out + offset);
    }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

// =============================================================================
// 5. Tag types + chain-shape traits
// =============================================================================

/// Marks an element that wraps `copy_tile` (CB → DEST). Pipeline treats these
/// specially — their FPU clash is OK to absorb because the post-loop reinit
/// covers the only effect (unpack MOP).
struct CopyTileTag {};

template <typename T>
constexpr bool is_copy_tile_op_v = std::is_base_of_v<CopyTileTag, T>;

// =============================================================================
// 6. CopyTile policy + reconfig
// =============================================================================

/**
 * Six wait/pop shapes. Each is a real workload pattern (lessons §2.1):
 *
 *   WaitAndPop          per-tile wait + per-tile pop      — streaming default
 *   WaitNoPop           per-tile wait + no pop            — fan-out first / persistent
 *   NoWaitPop           no wait       + per-tile pop      — fan-out last / pre-waited single
 *   NoWaitNoPop         no wait       + no pop            — caller manages
 *   WaitUpfrontPopAtEnd upfront wait  + upfront pop       — block access with indexed copy_tile
 *   CumulativeWait      wait(base+i)  + caller pops       — producer-aligned wait, prior tiles persist
 *
 * `cb_wait_front` is idempotent so `WaitNoPop + WaitAndPop` would also work
 * for fan-out — but `WaitNoPop + NoWaitPop` is one fewer NoC roundtrip and the
 * "first already waited" relationship is searchable at the call site.
 */
enum class CopyTilePolicy {
    WaitAndPop,
    WaitNoPop,
    NoWaitPop,
    NoWaitNoPop,
    WaitUpfrontPopAtEnd,
    CumulativeWait,
};

/// Whether CopyTile reconfigures srca on `init`. Parity with
/// `DestReuseReconfig` (lessons §2.4). Never bool.
enum class CopyTileReconfig { None, Input };

// =============================================================================
// 7. CopyTile element — one CB tile → one DEST slot
// =============================================================================

namespace detail {
struct EltwisePipelineDetail;  // friend
}

template <
    uint32_t CB,
    Dst Slot,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CopyTileReconfig Reconfig = CopyTileReconfig::None>
struct CopyTile : CopyTileTag {
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr CopyTilePolicy policy = Policy;
    static constexpr CopyTileReconfig reconfig = Reconfig;

    static constexpr bool is_upfront = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr bool is_cumulative = (Policy == CopyTilePolicy::CumulativeWait);
    static constexpr bool clashes_with_fpu = true;  // copy_tile programs unpack MOP

    static_assert(dst_idx < DST_HW_CEILING, "DEST slot exceeds hw ceiling (16)");

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }

    /// Pipeline-only — bumps the cumulative / upfront tile pointer.
    /// Public-but-private-by-friend: only `EltwisePipelineDetail` may call.
    ALWI void _pipeline_advance() const;
    ALWI void _pipeline_reset() const;

private:
    mutable uint32_t cb_tile_idx_ = 0;
    friend struct detail::EltwisePipelineDetail;
};

// =============================================================================
// 8. Core elements: FillScalar, FillConst, CopyDest
// =============================================================================

/// Runtime-valued fill — `value` lives in the chain element instance.
template <Dst Slot = Dst::D0>
struct FillScalar : UnaryOp<FillScalar<Slot>, Slot> {
    float value;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/// Compile-time-valued fill via bitcast (e.g. infinity, NaN).
template <uint32_t Bits, Dst Slot = Dst::D0>
struct FillConst : UnaryOp<FillConst<Bits, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/// DEST-to-DEST copy (`copy_dest_values`). Used for save-before-clobber
/// patterns (e.g. gelu_backward stashes tanh result).
template <Dst Src, Dst Dest, DataFormat DF = DataFormat::Float16_b>
struct CopyDest {
    static constexpr uint32_t in0 = static_cast<uint32_t>(Src);
    static constexpr uint32_t out = static_cast<uint32_t>(Dest);
    static constexpr uint32_t max_dst = (in0 > out ? in0 : out);
    static_assert(in0 < DST_HW_CEILING, "CopyDest Src exceeds hw ceiling (16)");
    static_assert(out < DST_HW_CEILING, "CopyDest Dest exceeds hw ceiling (16)");

    static constexpr bool is_upfront = false;
    static constexpr bool is_cumulative = false;
    static constexpr bool clashes_with_fpu = false;  // SFPU-side, no FPU clash
    static constexpr uint32_t cb = 0;

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }
};

// =============================================================================
// 9. EltwiseChain<Ops...> — variadic combinator, no compaction
// =============================================================================

template <typename... Ops>
struct EltwiseChain;

template <>
struct EltwiseChain<> {
    static constexpr uint32_t max_dst = 0;
    static constexpr uint32_t stride = 1;

    ALWI void apply(uint32_t = 0) const {}
    ALWI void apply_init_only() const {}
    ALWI void apply_exec_only(uint32_t = 0) const {}
    ALWI void wait_upfront(uint32_t /*num_tiles*/) const {}
    ALWI void pop_upfront(uint32_t /*num_tiles*/) const {}
    ALWI void advance_cumulative() const {}
    ALWI void reset_cumulative_and_upfront() const {}
};

template <typename First, typename... Rest>
struct EltwiseChain<First, Rest...> {
    First first;
    EltwiseChain<Rest...> rest;

    static constexpr uint32_t _rest_max = EltwiseChain<Rest...>::max_dst;
    static constexpr uint32_t max_dst = (First::max_dst > _rest_max ? First::max_dst : _rest_max);
    static constexpr uint32_t stride = max_dst + 1;

    constexpr EltwiseChain() = default;
    constexpr EltwiseChain(First f, Rest... r) : first(f), rest(r...) {}

    ALWI void apply(uint32_t offset = 0) const;
    ALWI void apply_init_only() const;
    ALWI void apply_exec_only(uint32_t offset = 0) const;
    ALWI void wait_upfront(uint32_t num_tiles) const;
    ALWI void pop_upfront(uint32_t num_tiles) const;
    ALWI void advance_cumulative() const;
    ALWI void reset_cumulative_and_upfront() const;
};

// =============================================================================
// 10. Chain-shape traits — used by the pipeline to pick init-hoist behavior.
// =============================================================================

namespace detail {

template <typename Chain>
struct ChainHasAnyCopyTile {
    static constexpr bool value = false;
};
template <typename First, typename... Rest>
struct ChainHasAnyCopyTile<EltwiseChain<First, Rest...>> {
    static constexpr bool value = is_copy_tile_op_v<First> || ChainHasAnyCopyTile<EltwiseChain<Rest...>>::value;
};

template <typename Chain>
struct ChainHasNonCopyTileFpuClash {
    static constexpr bool value = false;
};
template <typename First, typename... Rest>
struct ChainHasNonCopyTileFpuClash<EltwiseChain<First, Rest...>> {
    static constexpr bool value = (!is_copy_tile_op_v<First> && First::clashes_with_fpu) ||
                                  ChainHasNonCopyTileFpuClash<EltwiseChain<Rest...>>::value;
};

/// Check `cb_query` against any other upfront element in the rest of the chain.
template <uint32_t cb_query, typename Chain>
struct ChainHasUpfrontCB {
    static constexpr bool value = false;
};
template <uint32_t cb_query, typename First, typename... Rest>
struct ChainHasUpfrontCB<cb_query, EltwiseChain<First, Rest...>> {
    static constexpr bool value =
        (First::is_upfront && First::cb == cb_query) || ChainHasUpfrontCB<cb_query, EltwiseChain<Rest...>>::value;
};

template <typename Chain>
struct ChainHasDuplicateUpfrontCBs {
    static constexpr bool value = false;
};
template <typename First, typename... Rest>
struct ChainHasDuplicateUpfrontCBs<EltwiseChain<First, Rest...>> {
    static constexpr bool value = (First::is_upfront && ChainHasUpfrontCB<First::cb, EltwiseChain<Rest...>>::value) ||
                                  ChainHasDuplicateUpfrontCBs<EltwiseChain<Rest...>>::value;
};

/// First CopyTile CB in the chain (for srca reconfig); 0 if none.
template <typename Chain>
struct FirstCopyTileCB {
    static constexpr uint32_t value = 0;
};
template <typename First, typename... Rest>
struct FirstCopyTileCB<EltwiseChain<First, Rest...>> {
    static constexpr uint32_t value =
        is_copy_tile_op_v<First> ? First::cb : FirstCopyTileCB<EltwiseChain<Rest...>>::value;
};

}  // namespace detail

template <typename Chain>
constexpr bool chain_has_any_copy_tile_v = detail::ChainHasAnyCopyTile<Chain>::value;

template <typename Chain>
constexpr bool chain_has_non_copy_tile_fpu_clash_v = detail::ChainHasNonCopyTileFpuClash<Chain>::value;

template <typename Chain>
constexpr bool chain_has_duplicate_upfront_cbs_v = detail::ChainHasDuplicateUpfrontCBs<Chain>::value;

// =============================================================================
// 11. eltwise_chain(...) factory
// =============================================================================

/// Build an EltwiseChain from a variadic op list. No compaction, no
/// reordering. Compile-time aborts on duplicate upfront CBs.
template <typename... Ops>
constexpr ALWI auto eltwise_chain(Ops... ops) {
    using Chain = EltwiseChain<Ops...>;
    static_assert(
        !chain_has_duplicate_upfront_cbs_v<Chain>,
        "Two upfront CopyTile elements share a CB — split into separate CBs or "
        "use a single WaitUpfrontPopAtEnd element with multiple slots.");
    return Chain(ops...);
}

// =============================================================================
// 12. eltwise_pipeline output policy + entry point
// =============================================================================

enum class EltwiseOutputPolicy { PerTile, Bulk };

enum class EltwiseDataFormatReconfig { NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT };

/**
 * Per-tile eltwise pipeline.
 *
 * Steps per call:
 *   1. (compile-time) optional srca reconfig from first CopyTile's CB; pack
 *      reconfig from `ocb`.
 *   2. (compile-time) `copy_tile_to_dst_init_short(first_copy_tile_cb)` once.
 *   3. Upfront-wait pass — every is_upfront / WaitUpfrontPopAtEnd element
 *      runs `cb_wait_front(cb, num_tiles)` once.
 *   4. Per-tile loop:
 *        tile_regs_acquire();
 *        chain.apply(0);                 // init+exec each element every tile
 *        tile_regs_commit(); tile_regs_wait();
 *        cb_reserve_back(ocb, 1);
 *        pack_tile(static_cast<uint32_t>(pack_slot), ocb);
 *        cb_push_back(ocb, 1);
 *        tile_regs_release();
 *        chain.advance_cumulative();     // bump cumulative / upfront pointers
 *   5. Upfront-pop pass — every is_upfront element runs
 *      `cb_pop_front(cb, num_tiles)`. Cumulative elements do NOT pop;
 *      caller is responsible.
 *   6. Reset every CopyTile's `cb_tile_idx_` to 0.
 *
 * v1 takes the safe init-hoist path (init each element every tile). The
 * `apply_exec_only` shortcut is wired but currently unused — added for
 * future perf work driven by 2d measurements (lessons §4.2).
 */
template <
    EltwiseOutputPolicy OutPolicy = EltwiseOutputPolicy::PerTile,
    EltwiseDataFormatReconfig Reconfig = EltwiseDataFormatReconfig::INPUT_AND_OUTPUT,
    typename Chain>
ALWI void eltwise_pipeline(Chain chain, uint32_t ocb, uint32_t num_tiles, Dst pack_slot = Dst::D0);

}  // namespace compute_kernel_lib::eltwise

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
