// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_binary.hpp
 * @brief FPU binary ops (ELWADD / ELWSUB / ELWMUL) with broadcast,
 *        DestReuseOp chain element, and SFPU binary chain elements.
 *
 * Lives in `compute_kernel_lib::eltwise`. Independent of the legacy
 * `binary_op_helpers.hpp` — uses raw LLK calls only.
 *
 * PREREQUISITE: caller invokes `binary_op_init_common(icb_a, icb_b, ocb)`
 * once at the start of the kernel before calling `binary_op` /
 * `add` / `sub` / `mul` / `square`.
 *
 * ── Broadcast reference (lessons §5.3 — kept here so callers don't go hunting)
 *
 * | BroadcastDim | B shape  | B tile count | Companion reduce |
 * |--------------|----------|--------------|------------------|
 * | NONE         | [Ht,Wt]  | Ht*Wt        | —                |
 * | ROW          | [1,Wt]   | Wt           | REDUCE_COL       |
 * | COL          | [Ht,1]   | Ht           | REDUCE_ROW       |
 * | SCALAR       | [1,1]    | 1            | REDUCE_SCALAR    |
 *
 * ── Examples ─────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"
 *   using namespace compute_kernel_lib::eltwise;
 *
 *   // Caller-side init (once per kernel)
 *   binary_op_init_common(cb_a, cb_b, cb_out);
 *
 *   // 1. Element-wise add (streaming default)
 *   add(cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 2. Subtract per-row mean (mean is COL-shaped from REDUCE_ROW)
 *   sub<BroadcastDim::COL>(cb_in, cb_mean, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 3. Same-CB mul = square (helper dedups wait/pop internally)
 *   mul(cb_x, cb_x, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *   square(cb_x, cb_out, BinaryInputBlockShape::of(Ht, Wt));     // alias
 *
 *   // 4. Persistent B (e.g. softmax: subtract row max then reuse input)
 *   sub<BroadcastDim::COL,
 *       BinaryInputPolicy::WaitAndPopPerTile,
 *       BinaryInputPolicy::WaitUpfrontNoPop>(
 *       cb_in, cb_max, cb_centered, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 5. Caller-managed CB lifecycle (pre-loaded sharded inputs)
 *   cb_wait_front(cb_a, total); cb_wait_front(cb_b, total);
 *   add<BroadcastDim::NONE,
 *       BinaryInputPolicy::NoWaitNoPop,
 *       BinaryInputPolicy::NoWaitNoPop>(
 *       cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *   cb_pop_front(cb_a, total); cb_pop_front(cb_b, total);
 *
 *   // 6. PostOp chain — apply rsqrt to each output tile while it sits in DEST
 *   mul(cb_in, cb_in, cb_out, BinaryInputBlockShape::of(Ht, Wt),
 *       eltwise_chain(Rsqrt<>{}));
 *
 *   // 7. PostOp chain that loads a third CB — drives clashes_with_fpu reinit
 *   add(cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt),
 *       eltwise_chain(CopyTile<cb_c, Dst::D1>{}, SfpuMul<Dst::D0, Dst::D1, Dst::D0>{}));
 *
 *   // 8. DestReuseOp inside a PostOp chain — (a-b)*scale, scale persists
 *   sub(cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt),
 *       eltwise_chain(DestReuseMul<cb_scale>{}));
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

// =============================================================================
// 1. Op type + reuse type + reconfig knobs
// =============================================================================

enum class BinaryOpType { ADD, SUB, MUL };

/// Direction of DEST → SRC reuse.
enum class DestReuseType { ToSrcA, ToSrcB };

/// Whether DestReuseOp re-runs srca/srcb format reconfig on init.
enum class DestReuseReconfig { None, Input };

enum class BinaryDataFormatReconfig { NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT };

/**
 * CB-side wait/pop policy for `DestReuseOp`. Independent axis from `Reuse`.
 *
 *   WaitNoPop   — wait per tile, no pop          [default; batch_norm-style
 *                                                 persistent CB, caller pops once]
 *   WaitAndPop  — wait per tile, pop per tile    [streaming, e.g. mish bf16
 *                                                 where cb_input feeds one tile per iter]
 *   NoWaitPop   — skip wait, pop per tile        [paired with a prior CopyTile<WaitNoPop>
 *                                                 on the same CB — chain-internal handoff]
 *   NoWaitNoPop — skip wait, no pop              [caller fully manages]
 *
 * Mirrors `CopyTilePolicy`'s shape but stops at four corners — DestReuseOp
 * cannot batch (its FPU init reprograms unpack MOP each call), so the
 * upfront / cumulative variants are not exposed.
 */
enum class DestReuseInputPolicy { WaitNoPop, WaitAndPop, NoWaitPop, NoWaitNoPop };

// =============================================================================
// 2. Input / output policies
// =============================================================================

/**
 * Independent A/B policies (lessons §2.3). Each side picks its own
 * wait + pop shape.
 */
enum class BinaryInputPolicy {
    WaitAndPopPerTile,    // streaming default
    WaitUpfrontNoPop,     // pre-loaded, persists for re-use
    WaitUpfrontPopAtEnd,  // pre-loaded, helper pops bulk at end
    NoWaitNoPop,          // caller manages wait + pop
    NoWaitPopAtEnd,       // caller manages wait, helper pops bulk at end
};

enum class BinaryOutputPolicy { PerTile, Bulk };

// =============================================================================
// 3. Block shape — Ht / Wt in tiles
// =============================================================================

struct BinaryInputBlockShape {
    uint32_t rows;  // Ht (tiles)
    uint32_t cols;  // Wt (tiles)

    static constexpr BinaryInputBlockShape of(uint32_t r, uint32_t c) { return {r, c}; }
    static constexpr BinaryInputBlockShape single() { return {1, 1}; }
    static constexpr BinaryInputBlockShape row(uint32_t c) { return {1, c}; }
    static constexpr BinaryInputBlockShape col(uint32_t r) { return {r, 1}; }
};

// =============================================================================
// 4. PostOp gate — only EltwiseChain accepted
// =============================================================================

namespace detail {

template <typename T>
struct IsEltwiseChain : std::false_type {};
template <typename... Ops>
struct IsEltwiseChain<EltwiseChain<Ops...>> : std::true_type {};

template <typename T>
constexpr bool is_eltwise_chain_v = IsEltwiseChain<T>::value;

}  // namespace detail

// =============================================================================
// 5. DestReuseOp — chain element that runs A op DEST in-place.
//
// CB feeds either srca or srcb (per `ReuseType`); the existing DEST tile takes
// the other side. Drives `clashes_with_fpu` because it reprograms unpack MOP
// + binary FPU init.
// =============================================================================

template <
    uint32_t CB,
    BinaryOpType OpType,
    DestReuseType Reuse = DestReuseType::ToSrcB,
    Dst Slot = Dst::D0,
    DestReuseInputPolicy Policy = DestReuseInputPolicy::WaitNoPop,
    DestReuseReconfig Reconfig = DestReuseReconfig::None>
struct DestReuseOp {
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;

    static constexpr bool is_upfront = false;
    static constexpr bool is_cumulative = false;
    static constexpr bool clashes_with_fpu = true;

    static_assert(dst_idx < DST_HW_CEILING, "DestReuseOp slot exceeds hw ceiling (16)");

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }
};

/// Convenience aliases (lessons §6 — first generalization driver).
template <
    uint32_t CB,
    Dst Slot = Dst::D0,
    DestReuseInputPolicy P = DestReuseInputPolicy::WaitNoPop,
    DestReuseReconfig R = DestReuseReconfig::None>
using DestReuseAdd = DestReuseOp<CB, BinaryOpType::ADD, DestReuseType::ToSrcB, Slot, P, R>;

template <
    uint32_t CB,
    Dst Slot = Dst::D0,
    DestReuseInputPolicy P = DestReuseInputPolicy::WaitNoPop,
    DestReuseReconfig R = DestReuseReconfig::None>
using DestReuseSub = DestReuseOp<CB, BinaryOpType::SUB, DestReuseType::ToSrcB, Slot, P, R>;

template <
    uint32_t CB,
    Dst Slot = Dst::D0,
    DestReuseInputPolicy P = DestReuseInputPolicy::WaitNoPop,
    DestReuseReconfig R = DestReuseReconfig::None>
using DestReuseMul = DestReuseOp<CB, BinaryOpType::MUL, DestReuseType::ToSrcB, Slot, P, R>;

// =============================================================================
// 6. binary_op<...> — top-level FPU binary entry point
// =============================================================================

/**
 * Element-wise binary op A ⊕ B over a [Ht, Wt] tile block, with optional
 * BroadcastDim, independent A/B policies, output policy, dtype-reconfig knob,
 * and an `EltwiseChain` PostOp that runs on each output tile while it sits
 * in DEST.
 *
 * Same-CB inputs (icb_a == icb_b at runtime) dedup wait/pop internally
 * (lessons §3.6). Callers can pass the same CB twice to compute squares /
 * self-products without an alias.
 */
template <
    BinaryOpType OpType,
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy APolicy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy BPolicy = APolicy,
    BinaryOutputPolicy OutPolicy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = EltwiseChain<>>
ALWI void binary_op(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {});

// =============================================================================
// 7. Convenience aliases — add / sub / mul / square
// =============================================================================

template <
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy APolicy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy BPolicy = APolicy,
    BinaryOutputPolicy OutPolicy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = EltwiseChain<>>
ALWI void add(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {});

template <
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy APolicy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy BPolicy = APolicy,
    BinaryOutputPolicy OutPolicy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = EltwiseChain<>>
ALWI void sub(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {});

template <
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy APolicy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy BPolicy = APolicy,
    BinaryOutputPolicy OutPolicy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = EltwiseChain<>>
ALWI void mul(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {});

/// `square(cb, ocb, shape, post)` ≡ `mul(cb, cb, ocb, shape, post)`.
template <
    BinaryInputPolicy Policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy OutPolicy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig Reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = EltwiseChain<>>
ALWI void square(uint32_t icb, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {});

// =============================================================================
// 8. Binary SFPU chain elements (operate on two DEST slots, not CBs)
// =============================================================================

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuAdd : BinaryOp<SfpuAdd<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuSub : BinaryOp<SfpuSub<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMul : BinaryOp<SfpuMul<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuDiv : BinaryOp<SfpuDiv<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMax : BinaryOp<SfpuMax<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMin : BinaryOp<SfpuMin<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

}  // namespace compute_kernel_lib::eltwise

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.inl"
