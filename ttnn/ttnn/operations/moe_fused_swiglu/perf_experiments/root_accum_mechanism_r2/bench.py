# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ROUND 2 isolated bake-off: moe_fused_swiglu's ROOT cross-column reduce-accumulate mechanism.

WHAT CHANGED VS ROUND 1 (`perf_experiments/reduce_accum_mechanism/`)
-------------------------------------------------------------------
Round 1's honest baseline was ``add<input(cb), input(cb), output(cb)>(EltwiseShape::tiles(n))`` —
the DEFAULT per-tile lifecycle, which `eltwise_chain` silently clamps to ``block_size = 1``
(`eltwise_chain.inl:3054`). Perf 1 then GRADUATED `ELTWISE_BLK = 8` into the op, so the op's
current spelling is ``add<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), blk_out(cb_gate_acc)>
(blk_shape(gu_block_tiles))`` with `PerChunk` lifecycles + `OperandKind::Block`. Round 1's 1.22x /
1.35x was therefore measured against a baseline the op NO LONGER RUNS, and most of that win is
already banked. This bench re-baselines against the shipped spelling (VARIANT_BASELINE) and keeps
round 1's clamped spelling as VARIANT_PERTILE so the two rounds stitch together.

STRUCTURE — this is the ROOT's reduce loop, not a generic accumulate
-------------------------------------------------------------------
Per M-block a root holds its OWN local partial in `cb_gate_acc` / `cb_up_acc` (put there by the
gate/up matmul — it is NOT copied in, so there is no "seed" pass), then folds up to `fan_in` remote
children in, alternating the two roles per child exactly as
`moe_fused_swiglu_compute.cpp`'s `compute_reduce` does:

    for c in children:  add(gate_acc += reduce_gate_in[c]);  add(up_acc += reduce_up_in[c])

The bench reproduces that: the accumulators are PRE-FILLED resident sharded tensors exposed once
with `cb_reserve_back` + `cb_push_back` (no DMA — transport is a different assigned part), the
children are pre-resident too, and `ROLES = 2` interleaves gate/up per child. The interleave is
load-bearing for the in-place variants: two independent in-place chains alternating on different
CBs partially hide each other's producer/consumer stalls, so a single-role bench would OVERSTATE
the in-place penalty relative to the op.

`REPEATS` re-runs the whole per-M-block fold in one launch (children are re-exposed each pass), so
the constant launch/teardown cost can be differenced out: per-M-block cost = (T(R) - T(1))/(R - 1).

`FAN_IN` goes up to 10 so the same menu can be measured in the regime a two-phase REDUCE-SCATTER
restructure would create (each of the KGROUPS = 10 column cores folds ALL 10 contributors over a
1/10 slice of the block — many passes over few tiles, instead of few passes over many tiles).

THE MENU
--------
  0 baseline        the op's CURRENT approach verbatim: in-place blocked `add` per child, bfp8_b
                    accumulator. `acc` round-trips L1 (unpack acc + unpack child + FPU add + pack
                    acc, bfp8-requantized) once PER CHILD. `rmw` in
                    `examples/eltwise_l1_vs_dest_accumulate`'s ranking.
  1 pertile         round 1's baseline: identical math, DEFAULT per-tile lifecycle -> block clamped
                    to 1. Anchors this round against round 1 and prices what ELTWISE_BLK banked.
  2 pack_l1_acc     the PACKER folds each child onto the resident accumulator; `acc` is never
                    UNPACKED (1 unpack + 1 pack per tile per child instead of 2 + 1). Spelled
                    PURELY with helpers: `L1Accumulation::Enabled` + `TileOffset::Strided` (see the
                    kernel-head note — `Strided` is what keeps the pack index ADVANCING under L1
                    accumulation, which `walk` alone does not). VALID ONLY at a linear accumulator
                    format, so the accumulator CB becomes bfloat16 (+92,160 B of L1); the children
                    stay bfp8 on the wire and in L1. REDUCE_SLOTS stays 1.
  3 pack_l1_bfp8    the SAME mechanism onto the op's CURRENT bfp8_b accumulator — a CORRECTNESS
                    BUG, not a precision trade (the packer's L1-accumulate register does a linear
                    add, which is meaningless on a shared-exponent block-float tile). Measured here
                    only to re-confirm round 1's PCC 0.412 finding at the blocked spelling.
  4 dest_pair       TWO children co-resident (`REDUCE_SLOTS = 2`, +104,448 B): one DEST window
                    computes `acc + c[2k] + c[2k+1]` (`BinaryFpu` then `DestReuseBinary`) and packs
                    once — halving both the acc round-trips AND the pass count. bfp8 accumulator,
                    so NO format change and NO extra NoC payload.
  5 dest_full       ALL `fan_in` children co-resident (`REDUCE_SLOTS = fan_in`): one DEST window
                    per tile-block sums `acc + every child` and packs once. One pass total, one acc
                    round-trip total. Over the L1 budget at fan_in 4 / 48 tiles; measured as the
                    bfp8 ceiling (and it is IN budget on a reduce-scatter's small slices).
  6 pack_l1_full    ALL children co-resident AND a bf16 accumulator: DEST = `c0 + c1 (+ ...)` via
                    one `BinaryFpu` + `DestReuseBinary` chain, then ONE L1-accumulating pack folds
                    it onto `acc`, which is never unpacked at all. One pass, zero acc unpacks — the
                    absolute ceiling of this idea.
  7 pingpong        the in-place `add`'s output CB IS its input CB, so its `PerChunk`
                    reserve/push/wait/pop all target the SAME semaphore pair: the packer of window
                    w cannot reserve until the unpacker of window w has popped. This variant keeps
                    the same 2-unpack + 1-pack arithmetic but ALTERNATES between two accumulator
                    buffers (+104,448 B) so pack(w) and unpack(w+1) are free to overlap. It
                    isolates "is the baseline's cost the L1 traffic, or the self-dependency?" —
                    the answer decides whether options 2/4/5/6 could win for the reason we think.
  8 pack_l1_pair    THE WINNER, added after the first sweep priced the menu's two mechanisms
                    SEPARATELY. Fitting the sweep in units of one tile-unpack gives: plain pack 1,
                    L1-accumulating pack 2 (it is a read-modify-write in L1), `DestReuseBinary`
                    ~3.6 (the DEST->srcA transfer, exactly `examples/compute_fusion`'s "do NOT
                    reach for FPU dest-reuse just to skip L1" 0.82x finding). So the accumulator
                    unpack is worth removing and the dest-reuse is NOT worth paying — and the one
                    chain shape with BOTH properties is: sum TWO children with a single
                    `BinaryFpu` into DEST, then ONE L1-accumulating pack onto a bf16 accumulator.
                    2 unpacks + 1 (double-cost) pack per tile for TWO children = 4 units/2 children
                    against the baseline's 3 units/child. `REDUCE_SLOTS = 2` + bf16 accumulator, so
                    +196,608 B; the children stay bfp8, so the NoC payload is UNCHANGED.
"""

import ttnn

TILE = 32
MAX_CH = 10

VARIANT_BASELINE = 0
VARIANT_PERTILE = 1
VARIANT_PACK_L1_ACC = 2
VARIANT_PACK_L1_BFP8 = 3
VARIANT_DEST_PAIR = 4
VARIANT_DEST_FULL = 5
VARIANT_PACK_L1_FULL = 6
VARIANT_PINGPONG = 7
VARIANT_PACK_L1_PAIR = 8
VARIANT_PACK_L1_PAIR_ODDADD = 9

VARIANT_NAMES = {
    VARIANT_BASELINE: "baseline",
    VARIANT_PERTILE: "pertile(r1base)",
    VARIANT_PACK_L1_ACC: "pack_l1_acc",
    VARIANT_PACK_L1_BFP8: "pack_l1_bfp8",
    VARIANT_DEST_PAIR: "dest_pair",
    VARIANT_DEST_FULL: "dest_full",
    VARIANT_PACK_L1_FULL: "pack_l1_full",
    VARIANT_PINGPONG: "pingpong",
    VARIANT_PACK_L1_PAIR: "pack_l1_pair",
    VARIANT_PACK_L1_PAIR_ODDADD: "pk_pair_oddadd",
}

#: Accumulator dtype each variant REQUIRES. The children are bfp8_b in every variant — that is what
#: crosses the NoC today and this bench never changes it. `pack_l1_bfp8` deliberately asks for the
#: illegal combination so the bug is re-measured rather than asserted.
ACC_DTYPE = {
    VARIANT_BASELINE: ttnn.bfloat8_b,
    VARIANT_PERTILE: ttnn.bfloat8_b,
    VARIANT_PACK_L1_ACC: ttnn.bfloat16,
    VARIANT_PACK_L1_BFP8: ttnn.bfloat8_b,
    VARIANT_DEST_PAIR: ttnn.bfloat8_b,
    VARIANT_DEST_FULL: ttnn.bfloat8_b,
    VARIANT_PACK_L1_FULL: ttnn.bfloat16,
    VARIANT_PINGPONG: ttnn.bfloat8_b,
    VARIANT_PACK_L1_PAIR: ttnn.bfloat16,
    VARIANT_PACK_L1_PAIR_ODDADD: ttnn.bfloat16,
}


def reduce_slots_needed(variant, fan_in):
    """Concurrent child landing slots the variant needs in cb_reduce_*_in (the op's REDUCE_SLOTS).
    1 = the shipped one-child-at-a-time protocol, 0 extra L1."""
    if variant in (VARIANT_DEST_PAIR, VARIANT_PACK_L1_PAIR, VARIANT_PACK_L1_PAIR_ODDADD):
        return min(2, fan_in)
    if variant in (VARIANT_DEST_FULL, VARIANT_PACK_L1_FULL):
        return fan_in
    return 1


# CB ids: role A (gate) = acc 0, alt 1, children 2..11; role B (up) = acc 12, alt 13, children
# 14..23. Contiguous children let the kernel index them arithmetically (see the Seq fold).
CB_A_ACC, CB_A_ALT, CB_A_CH0 = 0, 1, 2
CB_B_ACC, CB_B_ALT, CB_B_CH0 = 12, 13, 14

_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

// moe_fused_swiglu ROOT reduce-accumulate, round-2 isolated bench. See bench.py's module docstring
// for the menu and the round-1 delta. Two notes that are load-bearing and NOT obvious from the
// helper surface:
//
// (1) L1-ACCUMULATING PACK THAT STILL ADVANCES ITS ADDRESS.
//     `PackTileImpl::walk` (eltwise_chain.inl:1008) is FALSE whenever `L1Accumulation != Disabled`,
//     so the plain `output(cb, ..., L1Accumulation::Enabled)` spelling PINS every write to
//     `out_idx = base` — a MANY:1 reduce, which is the wrong cardinality for this op's per-position
//     (48-tile-wide) accumulate. Round 1 worked around it by bypassing the field and driving the
//     raw `pack_reconfig_l1_acc(1)` register itself. That RAW-LLK BYPASS IS NOT NEEDED: `exec`'s
//     Strided leg (eltwise_chain.inl:1102-1103) is evaluated BEFORE `walk`, so
//     `TileOffset::Strided` + `StridedTileRange{0, N}` gives `out_idx = 0 + ht*N + wt + j`, which
//     at `Ht == 1` walks 0..N-1 exactly as the non-accumulating pack does — while
//     `out_of_order_output` is forced true by the L1-accumulation term, so `pack_tile` honors that
//     index instead of falling back to its own per-call-resetting internal counter. So this bench
//     uses NO raw LLK: every variant is pure kernel_lib, and the chain itself owns the
//     `pack_reconfig_l1_acc(1)` / `(0)` bracket (eltwise_chain.inl:3075, 3217) so hazard 2 (a
//     leaked packer-L1-accumulate register) cannot escape a chain call.
//
// (2) EVERY variant must be BLOCKED or the comparison is meaningless. `input(cb)` / `output(cb)`
//     default to per-TILE wait/pop/reserve/push, which makes `chain_supports_block_v` false and
//     silently clamps `block_size` to 1 (eltwise_chain.inl:3054). `blk_in` / `blk_out` below are
//     byte-identical to the op's own (moe_fused_swiglu_compute.cpp:112-114). VARIANT_PERTILE is the
//     ONLY variant that deliberately uses the clamped spelling — it is round 1's baseline, kept as
//     an anchor.
//
// CT args: [VARIANT, FAN_IN, N, BLK, REPEATS, ROLES, A_ACC, A_ALT, A_CH0, B_ACC, B_ALT, B_CH0]
// Child c of role A is CB (A_CH0 + c); of role B, (B_CH0 + c).

using namespace compute_kernel_lib;

constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t FAN_IN = get_compile_time_arg_val(1);
constexpr uint32_t N = get_compile_time_arg_val(2);
constexpr uint32_t BLK = get_compile_time_arg_val(3);
constexpr uint32_t REPEATS = get_compile_time_arg_val(4);
constexpr uint32_t ROLES = get_compile_time_arg_val(5);
constexpr uint32_t A_ACC = get_compile_time_arg_val(6);
constexpr uint32_t A_ALT = get_compile_time_arg_val(7);
constexpr uint32_t A_CH0 = get_compile_time_arg_val(8);
constexpr uint32_t B_ACC = get_compile_time_arg_val(9);
constexpr uint32_t B_ALT = get_compile_time_arg_val(10);
constexpr uint32_t B_CH0 = get_compile_time_arg_val(11);

static_assert(FAN_IN >= 1 && FAN_IN <= 10, "fan_in out of range");
static_assert(VARIANT <= 9, "variant out of range");
static_assert(ROLES == 1 || ROLES == 2, "roles must be 1 or 2");

// --- the op's own blocked-eltwise spelling, copied verbatim ---
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
ALWI auto blk_shape() { return EltwiseShape::tiles(N, BLK); }

// --- L1-accumulating, position-ADVANCING pack (note (1) above) ---
constexpr auto l1acc_out(uint32_t cb) {
    return output(
        cb,
        ReservePolicy::None,
        PushPolicy::None,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Enabled,
        DestAccumulation::Disabled,
        TileOffset::Strided);
}

template <uint32_t cb>
using L1AccPack = PackTile<l1acc_out(cb)>;
template <uint32_t cb>
using DR = DestReuseBinary<blk_in(cb), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>;

// Minimal index-sequence so the multi-child chains are written ONCE instead of unrolled by hand for
// every fan-in (a hand-unrolled 10-way if/else chain is where transcription bugs live).
template <uint32_t... Is>
struct Seq {};
template <uint32_t K, uint32_t... Is>
struct MakeSeq : MakeSeq<K - 1, K - 1, Is...> {};
template <uint32_t... Is>
struct MakeSeq<0, Is...> {
    using type = Seq<Is...>;
};
template <uint32_t K>
using SeqUpTo = typename MakeSeq<K>::type;

ALWI void expose(uint32_t cb) {
    cb_reserve_back(cb, N);
    cb_push_back(cb, N);
}

// Re-expose the resident children for one more pass of the fold (each fold pops them). Identical
// in every variant, so it never biases the comparison.
template <uint32_t... Is>
ALWI void expose_children_impl(Seq<Is...>) {
    (expose(A_CH0 + Is), ...);
    if constexpr (ROLES == 2) {
        (expose(B_CH0 + Is), ...);
    }
}
ALWI void expose_children() { expose_children_impl(SeqUpTo<FAN_IN>{}); }

// ---- per-child steps, gate/up interleaved exactly as compute_reduce does ----

template <uint32_t c>
ALWI void step_inplace_blocked() {
    add<blk_in(A_ACC), blk_in(A_CH0 + c), blk_out(A_ACC)>(blk_shape());
    if constexpr (ROLES == 2) {
        add<blk_in(B_ACC), blk_in(B_CH0 + c), blk_out(B_ACC)>(blk_shape());
    }
}

template <uint32_t c>
ALWI void step_inplace_pertile() {
    add<input(A_ACC), input(A_CH0 + c), output(A_ACC)>(EltwiseShape::tiles(N));
    if constexpr (ROLES == 2) {
        add<input(B_ACC), input(B_CH0 + c), output(B_ACC)>(EltwiseShape::tiles(N));
    }
}

template <uint32_t c>
ALWI void step_pack_l1_acc() {
    eltwise_chain(blk_shape(), CopyTile<blk_in(A_CH0 + c)>{}, L1AccPack<A_ACC>{StridedTileRange{0, N}});
    if constexpr (ROLES == 2) {
        eltwise_chain(blk_shape(), CopyTile<blk_in(B_CH0 + c)>{}, L1AccPack<B_ACC>{StridedTileRange{0, N}});
    }
}

// acc += c[k] + c[k+1] : the two children are summed by ONE FPU binary into DEST and the PACKER
// folds that onto the accumulator. THE POINT is what is ABSENT — no accumulator unpack (pack-L1
// accumulation) and no `DestReuseBinary` (the DEST->srcA transfer measured ~3.6x an unpack in the
// sweep, which is what caps `dest_full` / `pack_l1_full` at fan-in > 2). 2 unpacks + 1
// L1-accumulating pack per tile, for TWO children.
template <uint32_t k>
ALWI void step_pack_l1_pair() {
    eltwise_chain(
        blk_shape(),
        BinaryFpu<blk_in(A_CH0 + k), blk_in(A_CH0 + k + 1)>{},
        L1AccPack<A_ACC>{StridedTileRange{0, N}});
    if constexpr (ROLES == 2) {
        eltwise_chain(
            blk_shape(),
            BinaryFpu<blk_in(B_CH0 + k), blk_in(B_CH0 + k + 1)>{},
            L1AccPack<B_ACC>{StridedTileRange{0, N}});
    }
}

// acc = acc + c[k] + c[k+1] in ONE DEST window (2 children co-resident).
template <uint32_t k>
ALWI void step_dest_pair() {
    eltwise_chain(
        blk_shape(),
        BinaryFpu<blk_in(A_ACC), blk_in(A_CH0 + k)>{},
        DR<A_CH0 + k + 1>{},
        PackTile<blk_out(A_ACC)>{});
    if constexpr (ROLES == 2) {
        eltwise_chain(
            blk_shape(),
            BinaryFpu<blk_in(B_ACC), blk_in(B_CH0 + k)>{},
            DR<B_CH0 + k + 1>{},
            PackTile<blk_out(B_ACC)>{});
    }
}

// One in-place blocked add whose input/output accumulator CBs differ (ping-pong).
template <uint32_t inA, uint32_t outA, uint32_t inB, uint32_t outB, uint32_t c>
ALWI void step_pingpong() {
    add<blk_in(inA), blk_in(A_CH0 + c), blk_out(outA)>(blk_shape());
    if constexpr (ROLES == 2) {
        add<blk_in(inB), blk_in(B_CH0 + c), blk_out(outB)>(blk_shape());
    }
}

// ---- whole-fan-in single-pass folds (Rest = children 1..FAN_IN-1) ----

template <uint32_t... Rest>
ALWI void fold_dest_full_impl(Seq<Rest...>) {
    eltwise_chain(
        blk_shape(),
        BinaryFpu<blk_in(A_ACC), blk_in(A_CH0)>{},
        DR<A_CH0 + 1 + Rest>{}...,
        PackTile<blk_out(A_ACC)>{});
    if constexpr (ROLES == 2) {
        eltwise_chain(
            blk_shape(),
            BinaryFpu<blk_in(B_ACC), blk_in(B_CH0)>{},
            DR<B_CH0 + 1 + Rest>{}...,
            PackTile<blk_out(B_ACC)>{});
    }
}

// FAN_IN == 1 has no second CB operand for the FPU, so it degenerates to a plain copy into DEST
// plus the L1-accumulating pack (still zero accumulator unpacks). Overloading on the EMPTY pack
// keeps the multi-child expansion below from ever being instantiated with an empty tail.
ALWI void fold_pack_l1_full_impl(Seq<>) {
    eltwise_chain(blk_shape(), CopyTile<blk_in(A_CH0)>{}, L1AccPack<A_ACC>{StridedTileRange{0, N}});
    if constexpr (ROLES == 2) {
        eltwise_chain(blk_shape(), CopyTile<blk_in(B_CH0)>{}, L1AccPack<B_ACC>{StridedTileRange{0, N}});
    }
}

template <uint32_t R0, uint32_t... Rest>
ALWI void fold_pack_l1_full_impl(Seq<R0, Rest...>) {
    eltwise_chain(
        blk_shape(),
        BinaryFpu<blk_in(A_CH0), blk_in(A_CH0 + 1 + R0)>{},
        DR<A_CH0 + 1 + Rest>{}...,
        L1AccPack<A_ACC>{StridedTileRange{0, N}});
    if constexpr (ROLES == 2) {
        eltwise_chain(
            blk_shape(),
            BinaryFpu<blk_in(B_CH0), blk_in(B_CH0 + 1 + R0)>{},
            DR<B_CH0 + 1 + Rest>{}...,
            L1AccPack<B_ACC>{StridedTileRange{0, N}});
    }
}

// ---- compile-time unrolled child loops ----

template <uint32_t... Is>
ALWI void run_baseline(Seq<Is...>) {
    (step_inplace_blocked<Is>(), ...);
}
template <uint32_t... Is>
ALWI void run_pertile(Seq<Is...>) {
    (step_inplace_pertile<Is>(), ...);
}
template <uint32_t... Is>
ALWI void run_pack_l1_acc_children(Seq<Is...>) {
    (step_pack_l1_acc<Is>(), ...);
}
// PAIRS: Is enumerates 0..FAN_IN/2-1, child indices 2*Is and 2*Is+1 (after the odd leader, if any).
template <uint32_t off, uint32_t... Is>
ALWI void run_dest_pairs(Seq<Is...>) {
    (step_dest_pair<off + 2 * Is>(), ...);
}
template <uint32_t off, uint32_t... Is>
ALWI void run_pack_l1_pairs(Seq<Is...>) {
    (step_pack_l1_pair<off + 2 * Is>(), ...);
}
// PING-PONG pairs: each pair is acc -> alt -> acc, so the result always lands back in *_ACC.
template <uint32_t off, uint32_t... Is>
ALWI void run_pingpong_pairs(Seq<Is...>) {
    ((step_pingpong<A_ACC, A_ALT, B_ACC, B_ALT, off + 2 * Is>(),
      step_pingpong<A_ALT, A_ACC, B_ALT, B_ACC, off + 2 * Is + 1>()),
     ...);
}

void kernel_main() {
    compute_kernel_hw_startup(A_ACC, A_CH0, A_ACC);

    // The accumulators already hold this core's OWN local partial (the gate/up matmul put it there
    // in the real op) — expose the resident region once, no DMA. The pack-L1-accumulate variants
    // never READ the accumulator CB, so they manage it caller-side instead (below).
    constexpr bool acc_is_pack_only = (VARIANT == 2) || (VARIANT == 3) || (VARIANT == 6) || (VARIANT == 8) ||
                                      (VARIANT == 9);
    if constexpr (!acc_is_pack_only) {
        expose(A_ACC);
        if constexpr (ROLES == 2) {
            expose(B_ACC);
        }
    }

    // An odd fan-in folds its FIRST child with a plain in-place add, which makes the remaining
    // count even; that keeps `dest_pair`'s pairing and `pingpong`'s parity exact with no extra pass.
    constexpr uint32_t ODD = FAN_IN % 2;
    constexpr uint32_t PAIRS = (FAN_IN - ODD) / 2;

    for (uint32_t rep = 0; rep < REPEATS; ++rep) {
        expose_children();

        if constexpr (VARIANT == 0) {
            // ---- baseline: the op's CURRENT approach, verbatim (compute_reduce, gate/up
            // interleaved per child). `blk_out(cb) == blk_in(cb)` is the op's documented in-place
            // pattern: the chain's PerChunk pop precedes its PerChunk reserve, so the write pointer
            // trails the read pointer by exactly one chunk and each pass overlays the same tiles.
            run_baseline(SeqUpTo<FAN_IN>{});
        } else if constexpr (VARIANT == 1) {
            // ---- pertile: round 1's baseline. Same math; the default lifecycle clamps the DEST
            // window to ONE tile.
            run_pertile(SeqUpTo<FAN_IN>{});
        } else if constexpr (VARIANT == 2 || VARIANT == 3) {
            // ---- pack_l1_acc: one child unpacked per step, packer folds it onto the resident
            // accumulator. The accumulator is caller-managed for the whole M-block: reserved once
            // (so the pack base is the CB base), written FAN_IN times, published once, then
            // released — standing in for the downstream SwiGLU / bias consumer.
            CircularBuffer accA(A_ACC);
            CircularBuffer accB(B_ACC);
            accA.reserve_back(N);
            if constexpr (ROLES == 2) {
                accB.reserve_back(N);
            }
            run_pack_l1_acc_children(SeqUpTo<FAN_IN>{});
            accA.push_back(N);
            accA.wait_front(N);
            accA.pop_front(N);
            if constexpr (ROLES == 2) {
                accB.push_back(N);
                accB.wait_front(N);
                accB.pop_front(N);
            }
        } else if constexpr (VARIANT == 4) {
            // ---- dest_pair: two children per DEST window (REDUCE_SLOTS = 2). Halves both the pass
            // count and the accumulator round-trips; bfp8 accumulator, no format change.
            if constexpr (ODD == 1) {
                step_inplace_blocked<0>();
            }
            run_dest_pairs<ODD>(SeqUpTo<PAIRS>{});
        } else if constexpr (VARIANT == 5) {
            // ---- dest_full: every child co-resident, ONE pass, ONE accumulator round-trip.
            fold_dest_full_impl(SeqUpTo<FAN_IN - 1>{});
        } else if constexpr (VARIANT == 6) {
            // ---- pack_l1_full: every child co-resident AND a bf16 accumulator. The children are
            // summed in DEST and the packer folds that onto the accumulator, which is never
            // unpacked. One pass, zero accumulator unpacks.
            CircularBuffer accA(A_ACC);
            CircularBuffer accB(B_ACC);
            accA.reserve_back(N);
            if constexpr (ROLES == 2) {
                accB.reserve_back(N);
            }
            fold_pack_l1_full_impl(SeqUpTo<FAN_IN - 1>{});
            accA.push_back(N);
            accA.wait_front(N);
            accA.pop_front(N);
            if constexpr (ROLES == 2) {
                accB.push_back(N);
                accB.wait_front(N);
                accB.pop_front(N);
            }
        } else if constexpr (VARIANT == 9) {
            // ---- pack_l1_pair_oddadd: pack_l1_pair, except an ODD fan-in's leftover child is
            // folded with a PLAIN blocked in-place add instead of a solo CopyTile +
            // L1-accumulating pack. Costs one extra child-unpack but replaces a
            // read-modify-write pack (~2.2 unpacks) with a plain one, which is the better trade
            // whenever the stage is pack-bound — i.e. exactly at odd fan-in, where `pack_l1_pair`
            // leaves one unpaired pack behind. Identical to VARIANT 8 at even fan-in.
            //
            // The leftover add runs AFTER the caller-managed pack bracket is published, so the
            // accumulator CB is full when the in-place add wants to read it — no extra bookkeeping,
            // and the chain's own pack_reconfig_l1_acc(0) on exit means the plain pack that follows
            // cannot inherit a leaked packer-L1-accumulate register (hazard 2).
            CircularBuffer accA(A_ACC);
            CircularBuffer accB(B_ACC);
            accA.reserve_back(N);
            if constexpr (ROLES == 2) {
                accB.reserve_back(N);
            }
            run_pack_l1_pairs<ODD>(SeqUpTo<PAIRS>{});
            accA.push_back(N);
            if constexpr (ROLES == 2) {
                accB.push_back(N);
            }
            if constexpr (ODD == 1) {
                step_inplace_blocked<0>();
            }
            accA.wait_front(N);
            accA.pop_front(N);
            if constexpr (ROLES == 2) {
                accB.wait_front(N);
                accB.pop_front(N);
            }
        } else if constexpr (VARIANT == 8) {
            // ---- pack_l1_pair: TWO children per DEST window (REDUCE_SLOTS = 2) summed by one FPU
            // binary, then ONE L1-accumulating pack onto a bf16 accumulator. Zero accumulator
            // unpacks AND zero DestReuseBinary — the two costs the rest of the menu trades against
            // each other. An odd fan-in folds its leftover child with CopyTile + the same
            // L1-accumulating pack.
            CircularBuffer accA(A_ACC);
            CircularBuffer accB(B_ACC);
            accA.reserve_back(N);
            if constexpr (ROLES == 2) {
                accB.reserve_back(N);
            }
            if constexpr (ODD == 1) {
                step_pack_l1_acc<0>();
            }
            run_pack_l1_pairs<ODD>(SeqUpTo<PAIRS>{});
            accA.push_back(N);
            accA.wait_front(N);
            accA.pop_front(N);
            if constexpr (ROLES == 2) {
                accB.push_back(N);
                accB.wait_front(N);
                accB.pop_front(N);
            }
        } else {
            // ---- pingpong: identical arithmetic to the baseline, but the accumulator alternates
            // between two buffers so the in-place CB self-dependency is gone.
            if constexpr (ODD == 1) {
                step_inplace_blocked<0>();
            }
            run_pingpong_pairs<ODD>(SeqUpTo<PAIRS>{});
        }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(num_tiles):
    """One row of `num_tiles` tiles, height-sharded onto a single core (tiles row-major)."""
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def default_compute_kernel_config():
    """The op's FROZEN precision contract (moe_fused_swiglu.default_compute_kernel_config()),
    reconstructed here so every variant runs under the identical config and this bench never
    touches it: LoFi / approx SFPU / bf16 DEST (DEST_AUTO_LIMIT = 8 tiles) / precise bfp8 pack."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def child_cb(role, c):
    return (CB_A_CH0, CB_B_CH0)[role] + c


def acc_cb(role):
    return (CB_A_ACC, CB_B_ACC)[role]


def alt_cb(role):
    return (CB_A_ALT, CB_B_ALT)[role]


def make_operands(device, *, fan_in, block_tiles, acc_dtype, roles, variant, seed_val=0):
    """Build the resident operands for one measurement.

    Returns (tensors, torch_sources) where `tensors` maps CB id -> ttnn tensor and `torch_sources`
    is {"acc": [per-role fp32], "children": [[per-child fp32] per role]}.

    The ACCUMULATORS are pre-filled with this core's own local partial (as the gate/up matmul leaves
    them in the op). The CHILDREN are always bfloat8_b — that is what crosses the NoC in the shipped
    op. The ping-pong partner buffer is allocated for EVERY variant (a discarded `if constexpr`
    branch in a non-template function is still instantiated, so the kernel names that CB id in dead
    code regardless); it is far below any L1 pressure at these sizes, so it cannot bias a number.
    """
    import torch

    if fan_in > MAX_CH:
        raise ValueError(f"fan_in must be <= {MAX_CH}, got {fan_in}")

    torch.manual_seed(seed_val)
    cfg = create_sharded_memory_config(block_tiles)
    tensors = {}
    acc_src = []
    child_src = []

    for r in range(roles):
        local = torch.randn(TILE, block_tiles * TILE, dtype=torch.float32) * 0.1
        acc_src.append(local)
        tensors[acc_cb(r)] = ttnn.from_torch(
            local, dtype=acc_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg
        )
        tensors[alt_cb(r)] = ttnn.from_torch(
            torch.zeros_like(local), dtype=acc_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg
        )
        kids = []
        for c in range(fan_in):
            k = torch.randn(TILE, block_tiles * TILE, dtype=torch.float32) * 0.1
            kids.append(k)
            tensors[child_cb(r, c)] = ttnn.from_torch(
                k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg
            )
        child_src.append(kids)

    return tensors, {"acc": acc_src, "children": child_src}


def expected(torch_sources, role, repeats):
    """acc_final = local_partial + repeats * sum(children), in full fp32 (no intermediate
    quantization) — the SAME reference for every variant, so PCC prices each mechanism's own
    requantization cost."""
    import torch

    ref = torch_sources["acc"][role].clone()
    total = torch.zeros_like(ref)
    for k in torch_sources["children"][role]:
        total = total + k
    return ref + float(repeats) * total


def create_program_descriptor(tensors, *, variant, fan_in, block_tiles, blk, repeats, roles):
    compile_time_args = [
        variant,
        fan_in,
        block_tiles,
        blk,
        repeats,
        roles,
        CB_A_ACC,
        CB_A_ALT,
        CB_A_CH0,
        CB_B_ACC,
        CB_B_ALT,
        CB_B_CH0,
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        config=default_compute_kernel_config(),
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb, t) for cb, t in sorted(tensors.items())]
    io = [t for _, t in sorted(tensors.items())]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs), io


def run(tensors, *, variant, fan_in, block_tiles, blk, repeats, roles):
    desc, io = create_program_descriptor(
        tensors, variant=variant, fan_in=fan_in, block_tiles=block_tiles, blk=blk, repeats=repeats, roles=roles
    )
    return ttnn.generic_op(io, desc)


def acc_tensor(tensors, role):
    return tensors[acc_cb(role)]


def free(tensors):
    for t in tensors.values():
        ttnn.deallocate(t)


# ---------------------------------------------------------------------------
# L1 accounting for the menu (per CORE — CB sizes are uniform across all 110 cores)
# ---------------------------------------------------------------------------
BFP8_TILE_BYTES = 1088
BF16_TILE_BYTES = 2048
#: M_BLOCK * HN_PAD at the focus shape — the CB depth the op allocates for cb_gate_acc / cb_up_acc /
#: cb_reduce_gate_in / cb_reduce_up_in.
FOCUS_CB_TILES = 48


def l1_delta_bytes(variant, fan_in, cb_tiles=FOCUS_CB_TILES):
    """Extra L1 per core this variant needs against the shipped op (bfp8 accumulators, one child
    landing slot per role). Two role CB pairs (gate + up)."""
    delta = 0
    if ACC_DTYPE[variant] == ttnn.bfloat16:
        delta += 2 * cb_tiles * (BF16_TILE_BYTES - BFP8_TILE_BYTES)
    extra_slots = reduce_slots_needed(variant, fan_in) - 1
    if extra_slots > 0:
        delta += 2 * extra_slots * cb_tiles * BFP8_TILE_BYTES
    if variant == VARIANT_PINGPONG:
        acc_tile = BF16_TILE_BYTES if ACC_DTYPE[variant] == ttnn.bfloat16 else BFP8_TILE_BYTES
        delta += 2 * cb_tiles * acc_tile
    return delta
