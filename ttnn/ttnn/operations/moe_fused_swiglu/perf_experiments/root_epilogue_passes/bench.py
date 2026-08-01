# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only isolated bake-off: how many blocked DEST passes does the
moe_fused_swiglu reduce-ROOT epilogue need, and where do SiLU + the SwiGLU multiply belong?

Reconstructed from `kernels/moe_fused_swiglu_compute.cpp` (the `is_root && final_child` branch of
`compute_reduce` + the `compute_swiglu` block). It does NOT touch the real op.

The math every epilogue arm computes (focus shape m_eff=8, HN_PAD=6 -> 48 bfp8 tiles):

    h = SiLU(gate_acc + last_gate_child) * (up_acc + last_up_child)

SHIPPED (the `baseline` arm, post-Perf-1, ELTWISE_BLK=8):
  (a) `m_eff` SEPARATE `add_bias_bcast_rows<Elementwise, SubblockMajor, NoPostBias, SiluActivation>`
      calls, one per token tile-row, SiLU on the PACKER thread -> cb_gate_silu.
      The walk exists ONLY because the helper's bias index does not advance with `in0_subblock`
      (`bias_add_helpers.inl:141` for SubblockMajor, `:77` for TileRowMajor — BOTH read
      `bias_offset + in1_index_subblock_offset`, with no `in0_subblock` term), so an Elementwise
      bias spanning M_BLOCK tile-rows has to be walked one M-row at a time. Each call pays a full
      helper entry: srca/srcb/pack data-format reconfig + `add_tiles_init` + one 6-tile DEST window
      (pipeline fill/drain), for only HN_PAD = 6 tiles of work.
  (b) ONE blocked `add` (up_acc + last_up_child), 6 DEST windows of ELTWISE_BLK = 8 tiles.
  (c) ONE blocked FPU `mul` (cb_gate_silu * up_sum) -> cb_h_local.
  => 10 helper entries, 20 DEST windows, 2 intermediates in L1 (gate_silu, up_sum).

The arm menu collapses that in three orthogonal directions — pass COUNT, where SiLU RUNS, and how
SiLU is SPELLED on the SFPU — see ARMS below for the one-line description of each.

MEASURED HEADLINE (BH, 1 core, 48 tiles = the focus shape; full table in results.md). The epilogue is
NOT overhead-bound, it is SFPU-SiLU-bound:
    overhead (CB scaffolding only)              187 ns
    plain_add_x8 (8 x 48 plain tile-adds)    11,336 ns   ->  29.5 ns per bfp8 tile-add
    a_only_baseline (stage (a) alone)        48,192 ns   -> ~950 ns per SiLU tile
    baseline (the whole epilogue)            51,020 ns
So of the whole epilogue, ~46,800 ns (92 %) is 48 tiles of `silu_tile`, ~2,830 ns is the up-add plus
the SwiGLU multiply, and the ENTIRE per-helper-entry cost of the m_eff-call bias walk is ~850 ns
(~118 ns per entry). Collapsing 8 helper entries into 1 is therefore worth 1.7 % of stage (a); every
pass-count idea in this file is bounded by that. The only large lever is the SFPU spelling of SiLU:
`silu_tile` always runs the ACCURATE sigmoid, and the SFPLUT approx sigmoid measures 3.25x on the
whole epilogue (15,681 ns) for a worst-case -0.002 PCC.

Two kernel_lib gaps this bench had to work around locally (both are named in the returned report
so the coordinator can graduate them as helper additions rather than raw LLK):
  1. `eltwise_activations.hpp` exposes `Sigmoid` but NOT `Silu` as a chain element, even though its
     own file docstring claims it wraps `silu_tile`. Defined here as `SiluDest` in exactly the
     `Sigmoid` shape (4 lines). Everything else in every candidate arm is stock kernel_lib.
  2. `add_bias_bcast_rows` cannot walk a multi-row Elementwise bias in one call (the index bug
     above). The candidates side-step it with `eltwise_chain` instead of fixing the helper.

Precision contract (FROZEN, identical in every arm — never a lever): math_fidelity=LoFi,
math_approx_mode=True, fp32_dest_acc_en=False, dst_full_sync_en=False, bfp8_pack_precise=True,
every CB bfloat8_b (which is what the op uses for gate_acc / up_acc / reduce_*_in / gate_silu /
h_local). DEST_AUTO_LIMIT = 8 tiles (half-sync, 16-bit DEST).

NOTE on `math_approx_mode` and SiLU: `calculate_silu` (ckernel_sfpu_silu.h) hardcodes the ACCURATE
sigmoid — `silu_init` calls `sigmoid_init<false>()` and the body calls `_sfpu_sigmoid_` (exp_21f +
one reciprocal iteration) — so `silu_tile` IGNORES the user's math_approx_mode=True. `sigmoid_tile`
does honour it (`calculate_sigmoid<APPROXIMATION_MODE=true>` -> `calculate_sigmoid_appx`, a single
`lut(val,l0,l1,l2) + 0.5f` SFPLUT op with a 3-LReg load macro in its init). The `*_sigappx_*` arms
price that; their PCC is reported next to their ns and they are options WITH a precision cost, never
a silent substitution.

Implementation note (inherited from round 1's `root_epilogue_fusion/`, do not undo): ONE kernel
source PER ARM, no `if constexpr` method ladder in a single translation unit — unreached branches
still compile into the TU and corrupt the UNPACK<->MATH dest handshake, which hung the device.
`m_eff` / `hn_pad` / `kernel_iters` are baked in as literals.

Bench-vs-op deviations (deliberate, identical across arms, so every delta stays attributable):
  * The op's up-add is IN-PLACE (`add<blk_in(cb_up_acc), .., blk_out(cb_up_acc)>`); here it writes a
    separate `cb_up_sum` (same tile count, same formats, different L1 address) because an in-place
    chain on a tensor-backed CB is the pattern round 1's reduce-tree bench hung on.
  * The tensor-backed input CBs are re-marked (reserve/push/wait) once per kernel iteration so the
    loop can re-run; the `overhead` arm measures exactly that scaffolding so it can be subtracted.
"""

import ttnn

TILE = 32

# ---------------------------------------------------------------------------
# CB assignment
# ---------------------------------------------------------------------------
CB_GATE_ACC = 0  # tensor-backed: gate accumulator before the last child (bfp8_b)
CB_UP_ACC = 1  # tensor-backed: up accumulator before the last child (bfp8_b)
CB_REDUCE_GATE_IN = 2  # tensor-backed: last child's gate partial (bfp8_b)
CB_REDUCE_UP_IN = 3  # tensor-backed: last child's up partial (bfp8_b)
CB_GATE_SILU = 4  # scratch: SiLU(gate_sum)   — the 52,224 B the fused arms delete
CB_UP_SUM = 5  # scratch: up_acc + reduce_up_in (op does this in-place on cb_up_acc)
CB_GATE_SUM = 6  # scratch: gate_acc + reduce_gate_in (2-pass arms only)
CB_H_LOCAL = 16  # tensor-backed output: SiLU(gate_sum) * up_sum

BASELINE = "baseline"

# ---------------------------------------------------------------------------
# Kernel preamble — includes, CB ids, the op's own blk_in/blk_out spelling, and the two
# locally-defined chain elements (see the module docstring for why they are local).
# ---------------------------------------------------------------------------
_PREAMBLE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

using namespace ckernel;
using namespace compute_kernel_lib;

constexpr uint32_t cb_gate_acc = @CB_GATE_ACC@;
constexpr uint32_t cb_up_acc = @CB_UP_ACC@;
constexpr uint32_t cb_reduce_gate_in = @CB_REDUCE_GATE_IN@;
constexpr uint32_t cb_reduce_up_in = @CB_REDUCE_UP_IN@;
constexpr uint32_t cb_gate_silu = @CB_GATE_SILU@;
constexpr uint32_t cb_up_sum = @CB_UP_SUM@;
constexpr uint32_t cb_gate_sum = @CB_GATE_SUM@;
constexpr uint32_t cb_h_local = @CB_H_LOCAL@;

constexpr uint32_t M_EFF = @M_EFF@;
constexpr uint32_t HN_PAD = @HN_PAD@;
constexpr uint32_t KERNEL_ITERS = @ITERS@;
constexpr uint32_t BT = M_EFF * HN_PAD;              // gu_block_tiles
constexpr uint32_t ELTWISE_BLK = @BLK@;              // the op's graduated DEST window (Perf 1)

static_assert(HN_PAD <= compute_kernel_lib::DEST_AUTO_LIMIT, "one token tile-row must fit DEST");

// The op's own blocked-eltwise spelling (moe_fused_swiglu_compute.cpp:112-114). PerChunk+PerChunk
// with OperandKind::Block is the ONLY input policy that keeps `eltwise_chain` from silently
// clamping block_size to 1 (eltwise_chain.inl:1511 / :3054).
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
// Same walk, data-format reconfig SUPPRESSED. Legal here only because every CB in the epilogue is
// bfloat8_b, so the reconfig is pure wasted MMIO (examples/compute_block_size, second lever).
constexpr auto blk_in_nr(uint32_t cb) {
    return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block, DataFormatReconfig::Disabled);
}
constexpr auto blk_out_nr(uint32_t cb) {
    return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk, DataFormatReconfig::Disabled);
}
// Caller-managed window: no wait/pop/reserve/push inside the walk (None+None still supports a
// block, eltwise_chain.inl:1511). Used by the attribution arm and by the two-reader-one-CB arm.
constexpr auto held_in(uint32_t cb) { return input(cb, WaitPolicy::None, PopPolicy::None, OperandKind::Block); }
constexpr auto held_out(uint32_t cb) { return output(cb, ReservePolicy::None, PushPolicy::None); }

ALWI auto blk_shape(uint32_t n) { return EltwiseShape::tiles(n, ELTWISE_BLK); }

// A chain containing a `DestReuseBinary` must leave ONE DEST lane spare: measured wrong values at a
// window of 8 = DEST_AUTO_LIMIT, correct at 1..7 (see the DEST-reuse window rule in bench.py).
// Applied ONLY to the chains that contain one, so every other pass keeps the full ELTWISE_BLK.
constexpr uint32_t ELTWISE_BLK_DR = (ELTWISE_BLK < DEST_AUTO_LIMIT) ? ELTWISE_BLK : (DEST_AUTO_LIMIT - 1);
ALWI auto dr_shape(uint32_t n) { return EltwiseShape::tiles(n, ELTWISE_BLK_DR); }

namespace compute_kernel_lib {
// LOCAL chain element — kernel_lib's eltwise_activations.hpp has Sigmoid but no Silu (its own
// docstring claims silu_tile is wrapped; it is not). Identical shape to Sigmoid.
template <Dst Slot = Dst::D0>
struct SiluDest : UnaryOp<SiluDest<Slot>, Slot> {
    static ALWI void init() { silu_tile_init(); }
    static ALWI void exec_impl(uint32_t off) { silu_tile(to_u32(Slot) + off); }
};
// LOCAL chain element — kernel_lib's Sigmoid hardcodes fast_and_approx=false. This one exposes it,
// which is what makes the user's math_approx_mode=True reachable for the sigmoid LUT path.
template <bool Appx, Dst Slot = Dst::D0>
struct SigmoidDestA : UnaryOp<SigmoidDestA<Appx, Slot>, Slot> {
    static ALWI void init() { sigmoid_tile_init<Appx>(); }
    static ALWI void exec_impl(uint32_t off) { sigmoid_tile<VectorMode::RC, Appx>(to_u32(Slot) + off); }
};
}  // namespace compute_kernel_lib
"""

# Boot: mark every tensor-backed input CB available once.
_BOOT_CBS = r"""
    cb_reserve_back(cb_gate_acc, BT);
    cb_push_back(cb_gate_acc, BT);
    cb_reserve_back(cb_up_acc, BT);
    cb_push_back(cb_up_acc, BT);
    cb_reserve_back(cb_reduce_gate_in, BT);
    cb_push_back(cb_reduce_gate_in, BT);
    cb_reserve_back(cb_reduce_up_in, BT);
    cb_push_back(cb_reduce_up_in, BT);
    cb_wait_front(cb_gate_acc, BT);
    cb_wait_front(cb_up_acc, BT);
    cb_wait_front(cb_reduce_gate_in, BT);
    cb_wait_front(cb_reduce_up_in, BT);
"""

# Re-mark blocks (only what the arm actually popped may be re-marked, or reserve_back deadlocks).
_REMARK_GATE = r"""
            cb_reserve_back(cb_gate_acc, BT);
            cb_push_back(cb_gate_acc, BT);
            cb_wait_front(cb_gate_acc, BT);
            cb_reserve_back(cb_reduce_gate_in, BT);
            cb_push_back(cb_reduce_gate_in, BT);
            cb_wait_front(cb_reduce_gate_in, BT);
"""
_REMARK_UP = r"""
            cb_reserve_back(cb_up_acc, BT);
            cb_push_back(cb_up_acc, BT);
            cb_wait_front(cb_up_acc, BT);
            cb_reserve_back(cb_reduce_up_in, BT);
            cb_push_back(cb_reduce_up_in, BT);
            cb_wait_front(cb_reduce_up_in, BT);
"""
_DRAIN_OUT = r"""
            cb_wait_front(cb_h_local, BT);
            cb_pop_front(cb_h_local, BT);
"""

_BOOT_PACKER_SILU = (
    "    // Packer-thread SiLU init (sfpu_activation_helpers.hpp:157-174 — the activation rides\n"
    "    // the PACK thread, replacing tile_regs_wait, so it overlaps MATH).\n"
    "    ActivationInitHelper<KernelActivation::SILU>::init();"
)

# ---------------------------------------------------------------------------
# Stage snippets
# ---------------------------------------------------------------------------

# (a) SHIPPED: m_eff helper calls, packer SiLU, bias walked with bias_offset.
_A_BASELINE = r"""
        {
            CircularBuffer gate_buf(cb_gate_acc), rg_buf(cb_reduce_gate_in), silu_buf(cb_@ADEST@);
            rg_buf.wait_front(BT);
            for (uint32_t m = 0; m < M_EFF; ++m) {
                add_bias_bcast_rows<
                    BiasBroadcast::Elementwise,
                    OutputCBLayout::SubblockMajor,
                    bias_add_config::NoPostBias,
                    SiluActivation>(
                    gate_buf, rg_buf, silu_buf, BiasAddShape::of(1, 1, 1, HN_PAD), {}, m * HN_PAD);
            }
            rg_buf.pop_front(BT);
        }
"""

# (a) RAW LLK: ONE reconfig + ONE add_tiles_init hoisted out of the row walk, m_eff DEST windows of
# HN_PAD tiles, packer SiLU. This is round 1's `hoisted_bias` arm — the one that measured a further
# 1.07x-vs-1.05x and did NOT graduate.
_A_HOIST_ROWS = r"""
        // RAW LLK (bypasses add_bias_bcast_rows). Justification: the helper re-issues
        // reconfig_data_format_srca/srcb + pack_reconfig_data_format + add_tiles_init on EVERY call,
        // and its bias index cannot advance with in0_subblock (bias_add_helpers.inl:141), so the
        // shipped code pays M_EFF helper entries for M_EFF*HN_PAD tiles. Hoisting the setup out of
        // the walk keeps the identical DEST window shape (HN_PAD tiles/row, packer SiLU) and the
        // identical instruction sequence per row, isolating the per-entry setup cost alone.
        reconfig_data_format_srca(cb_gate_acc);
        reconfig_data_format_srcb(cb_reduce_gate_in);
        pack_reconfig_data_format(cb_@ADEST@);
        add_tiles_init(cb_gate_acc, cb_reduce_gate_in);
        cb_reserve_back(cb_@ADEST@, BT);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                add_tiles(cb_gate_acc, cb_reduce_gate_in, m * HN_PAD + c, m * HN_PAD + c, c);
            }
            tile_regs_commit();
            apply_activation_from_pack<KernelActivation::SILU>(HN_PAD);
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                pack_tile(c, cb_@ADEST@);
            }
            tile_regs_release();
        }
        cb_push_back(cb_@ADEST@, BT);
        cb_pop_front(cb_gate_acc, BT);
        cb_pop_front(cb_reduce_gate_in, BT);
"""

# (a) RAW LLK, ONE pass over the whole block in ELTWISE_BLK-tile DEST windows, packer SiLU. Same
# window width as every candidate chain, so this is the clean PACKER-vs-MATH SiLU A/B.
_A_BLK_PACKER = r"""
        // RAW LLK. Justification: no kernel_lib surface can put an SFPU activation on the PACK
        // thread inside a blocked eltwise walk — `apply_activation_from_pack` is reachable only
        // through matmul_block / add_bias_bcast_rows, both of which impose their own subblock
        // walk. This arm exists purely to price the packer-thread overlap at the SAME
        // ELTWISE_BLK window width the chain arms use.
        reconfig_data_format_srca(cb_gate_acc);
        reconfig_data_format_srcb(cb_reduce_gate_in);
        pack_reconfig_data_format(cb_@ADEST@);
        add_tiles_init(cb_gate_acc, cb_reduce_gate_in);
        cb_reserve_back(cb_@ADEST@, BT);
        for (uint32_t t = 0; t < BT; t += ELTWISE_BLK_DR) {
            const uint32_t w = (BT - t < ELTWISE_BLK_DR) ? (BT - t) : ELTWISE_BLK_DR;
            tile_regs_acquire();
            for (uint32_t c = 0; c < w; ++c) {
                add_tiles(cb_gate_acc, cb_reduce_gate_in, t + c, t + c, c);
            }
            tile_regs_commit();
            apply_activation_from_pack<KernelActivation::SILU>(w);
            for (uint32_t c = 0; c < w; ++c) {
                pack_tile(c, cb_@ADEST@);
            }
            tile_regs_release();
        }
        cb_push_back(cb_@ADEST@, BT);
        cb_pop_front(cb_gate_acc, BT);
        cb_pop_front(cb_reduce_gate_in, BT);
"""

# (a) ONE blocked helper chain: FPU add -> DEST, SiLU on the MATH thread in DEST, pack. 1 call.
_A_CHAIN = r"""
        eltwise_chain(
            blk_shape(BT),
            BinaryFpu<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), BinaryFpuOp::Add>{},
            SiluDest<>{},
            PackTile<blk_out(cb_@ADEST@)>{});
"""
_A_CHAIN_NR = r"""
        eltwise_chain(
            blk_shape(BT),
            BinaryFpu<blk_in_nr(cb_gate_acc), blk_in_nr(cb_reduce_gate_in), BinaryFpuOp::Add>{},
            SiluDest<>{},
            PackTile<blk_out_nr(cb_@ADEST@)>{});
"""

# (a) TWO blocked passes: ONE plain blocked add, then SiLU as its own blocked SFPU pass. Gives up
# the free packer-thread SiLU AND pays a 48-tile L1 round trip, in exchange for 2 calls not M_EFF.
_A_ADD_THEN_SILU = r"""
        add<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), blk_out(cb_gate_sum)>(blk_shape(BT));
        // held_in + a caller-owned wait/pop, NOT blk_in: the sigmoid twin below needs TWO readers on
        // this one CB, and only None+None / PerChunk+PerChunk keep eltwise_chain from clamping the
        // block to 1 (eltwise_chain.inl:1511). Spelled identically in both arms so the A/B is clean.
        cb_wait_front(cb_gate_sum, BT);
        eltwise_chain(
            @P2SHAPE@(BT),
            CopyTile<held_in(cb_gate_sum)>{},
            SiluDest<>{},
            PackTile<blk_out(cb_@ADEST@)>{});
        cb_pop_front(cb_gate_sum, BT);
"""

# (a) TWO blocked passes, SFPU spelled as sigmoid + a dest-reuse FPU multiply against the SAME
# staged sum. Same pass count and same L1 traffic as _A_ADD_THEN_SILU_CK, so the delta is PURELY the
# SFPU spelling: silu_tile (always-accurate sigmoid, x*sigmoid(x) inside ONE SFPU op) vs
# sigmoid_tile<., Appx> + an FPU multiply.
_A_ADD_TO_GATE_SUM = r"""
        add<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), blk_out(cb_gate_sum)>(blk_shape(BT));
"""
_A_SIGMOID_MUL = r"""
        cb_wait_front(cb_gate_sum, BT);
        eltwise_chain(
            dr_shape(BT),
            CopyTile<held_in(cb_gate_sum)>{},
            SigmoidDestA<@APPX@>{},
            DestReuseBinary<held_in(cb_gate_sum), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>{},
            PackTile<blk_out(cb_@ADEST@)>{});
        cb_pop_front(cb_gate_sum, BT);
"""

# The leanest SFPLUT-sigmoid shape: ONE fused pass carries BOTH multiplies as dest-reuse FPU ops
# (x*sigmoid(x) and then *up_sum), so cb_gate_silu never materializes and the epilogue is 3 passes.
# Two DestReuseBinary elements, one DEST lane, window 7.
_SIGAPPX_FUSED = r"""
        add<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), blk_out(cb_gate_sum)>(blk_shape(BT));
        add<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), blk_out(cb_up_sum)>(blk_shape(BT));
        cb_wait_front(cb_gate_sum, BT);
        eltwise_chain(
            dr_shape(BT),
            CopyTile<held_in(cb_gate_sum)>{},
            SigmoidDestA<true>{},
            DestReuseBinary<held_in(cb_gate_sum), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>{},
            DestReuseBinary<blk_in(cb_up_sum), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>{},
            PackTile<blk_out(cb_h_local)>{});
        cb_pop_front(cb_gate_sum, BT);
"""

# (b) the up add, and (c) the SwiGLU multiply — exactly as shipped (blocked, 1 call each).
_B_UP_ADD = r"""
        add<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), blk_out(cb_up_sum)>(blk_shape(BT));
"""
_C_MUL = r"""
        mul<blk_in(cb_gate_silu), blk_in(cb_up_sum), blk_out(cb_h_local)>(blk_shape(BT));
"""

# --- the DEST-reuse window rule (MEASURED, see test_diag_dest_reuse_pattern) --------------------
# Any eltwise_chain containing a `DestReuseBinary` element is CORRECT at a DEST window of <= 7 tiles
# and produces WRONG VALUES at 8 = DEST_AUTO_LIMIT: the dest-reuse LLK needs one spare DEST slot, and
# `chain_max_block_value()` (eltwise_chain.inl:1482-1493) subtracts that slot only on the
# `any_dest_accumulation` branch — the ordinary branch returns `DEST_AUTO_LIMIT / lane_width` = 8, so
# block_size 8 is admitted and silently overruns. Measured at 48 tiles, one call:
#   fuse_silu_mul PCC 0.999828 at blk 1..7  vs  0.972295 at blk 8
#   fuse_up_mul   PCC 0.999810 at blk 1..7  vs  0.912151 at blk 8
# So every dest-reuse arm below runs at ELTWISE_BLK = 7 (`blk_cap`), NOT 8. That one-lane haircut is
# part of the arm's price tag, and it is why the twin arms are also measured at 7.
_CHUNK_NOTE = r"""        // ONE eltwise_chain CALL PER DEST WINDOW instead of one call for the whole block. Prices the
        // alternative to the one-lane window haircut: re-emitting the chain setup per window. (An earlier
        // draft of this bench believed the DEST-reuse corruption was a hoisted-init defect; the
        // window-width sweep in test_diag_dest_reuse_pattern disproved that — per-window calls at
        // window 8 are still wrong, and a single call at window 7 is right.)"""


def _per_window(inner):
    return (
        _CHUNK_NOTE
        + r"""
        for (uint32_t t = 0; t < BT; t += ELTWISE_BLK_DR) {
            const uint32_t w = (BT - t < ELTWISE_BLK_DR) ? (BT - t) : ELTWISE_BLK_DR;
"""
        + inner
        + "        }\n"
    )


# (b)+(a)+(c) FUSED, order A (SFPU-then-FPU through DEST): stage the up sum in L1 first, then gate
# add -> SiLU -> dest-reuse FPU multiply -> pack h. cb_gate_silu NEVER MATERIALIZES.
_FUSE_SILU_MUL_CHAIN = r"""            eltwise_chain(
                @SHAPE@,
                BinaryFpu<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), BinaryFpuOp::Add>{},
                SiluDest<>{},
                DestReuseBinary<blk_in(cb_up_sum), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>{},
                PackTile<blk_out(cb_h_local)>{});
"""
_UP_SUM_LINE = r"""
        add<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), blk_out(cb_up_sum)>(blk_shape(BT));
"""
_FUSE_SILU_MUL = _UP_SUM_LINE + _FUSE_SILU_MUL_CHAIN.replace("@SHAPE@", "dr_shape(BT)")
_FUSE_SILU_MUL_PW = _UP_SUM_LINE + _per_window(_FUSE_SILU_MUL_CHAIN.replace("@SHAPE@", "EltwiseShape::tiles(w, w)"))

# Same fusion, order B (the round-1 shape): SiLU still round-trips through cb_gate_silu, the UP sum
# is what never materializes. In the op the up sum is in-place on cb_up_acc, so this frees no L1.
_FUSE_UP_MUL = r"""
        eltwise_chain(
            blk_shape(BT),
            BinaryFpu<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), BinaryFpuOp::Add>{},
            SiluDest<>{},
            PackTile<blk_out(cb_gate_silu)>{});
        eltwise_chain(
            dr_shape(BT),
            BinaryFpu<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), BinaryFpuOp::Add>{},
            DestReuseBinary<blk_in(cb_gate_silu), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>{},
            PackTile<blk_out(cb_h_local)>{});
"""

# ONE pass for the WHOLE epilogue: both adds live in DEST at once (lane width 2, so the chain
# clamps the window to DEST_AUTO_LIMIT/2 = 4 tiles), SiLU in DEST, and the SwiGLU multiply is an
# SFPU DEST-DEST mul. Zero intermediates in L1: 4 CB reads, 1 CB write, 1 helper entry.
_SINGLE_PASS = r"""
        eltwise_chain(
            blk_shape(BT),
            BinaryFpu<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), BinaryFpuOp::Add>{},
            BinaryFpu<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), BinaryFpuOp::Add, BroadcastDim::None, Dst::D1>{},
            SiluDest<>{},
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<blk_out(cb_h_local)>{});
"""

# Same single pass, SiLU spelled with the SFPLUT approx sigmoid. `x * sigmoid(x)` needs x twice, and
# DEST holds no spare copy — so the order is rearranged instead of duplicating a tile:
#   D1 = g*u  (SFPU mul, keeps g in D0), D0 = sigmoid_appx(g), D0 = D0 * D1 = silu(g)*u.
# Same function, different rounding order and an approximated sigmoid -> PCC is reported per arm.
_SINGLE_PASS_SIGAPPX = r"""
        eltwise_chain(
            blk_shape(BT),
            BinaryFpu<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), BinaryFpuOp::Add>{},
            BinaryFpu<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), BinaryFpuOp::Add, BroadcastDim::None, Dst::D1>{},
            MulBinary<Dst::D0, Dst::D1, Dst::D1>{},
            SigmoidDestA<true>{},
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<blk_out(cb_h_local)>{});
"""

# Attribution arm: the root's EIGHT plain 48-tile blocked reduce adds (384 bfp8 tile-adds), reading
# a held window so only the OUTPUT carries a CB lifecycle.
#
# The output MUST keep its per-chunk reserve/push (blk_out) and be drained between calls: `pack_tile`
# advances an internal per-CB write-tile pointer that only `cb_push_back` resets (api/compute/pack.h
# :107), so eight 48-tile passes with ReservePolicy::None walk 384 tiles past a 48-tile CB and
# trample the L1-resident tensors behind it. That was a real bug in this bench's first draft; it
# corrupted every arm dispatched afterwards.
_PLAIN_ADD_X8 = r"""
        for (uint32_t i = 0; i < 8; ++i) {
            add<held_in(cb_gate_acc), held_in(cb_reduce_gate_in), blk_out(cb_h_local)>(blk_shape(BT));
            if (i + 1 < 8) {
                cb_wait_front(cb_h_local, BT);
                cb_pop_front(cb_h_local, BT);
            }
        }
"""

# Scaffolding floor: the per-iteration CB bookkeeping every epilogue arm pays, and nothing else.
_OVERHEAD = r"""
        cb_pop_front(cb_gate_acc, BT);
        cb_pop_front(cb_reduce_gate_in, BT);
        cb_pop_front(cb_up_acc, BT);
        cb_pop_front(cb_reduce_up_in, BT);
        cb_reserve_back(cb_h_local, BT);
        cb_push_back(cb_h_local, BT);
"""


def _a_stage(snippet, dest):
    """Point stage (a) at cb_gate_silu (full-epilogue arms) or cb_h_local (stage-a-only arms)."""
    return snippet.replace("@ADEST@", dest)


# ---------------------------------------------------------------------------
# Arm table: body, boot init, remark blocks, reference kind, L1 scratch usage.
#   ref: "epilogue" -> silu(g0+g1)*(u0+u1) | "gate_silu" -> silu(g0+g1) | "add" -> g0+g1 | None
#   scratch: which scratch CBs the arm needs (drives the L1 delta column)
# ---------------------------------------------------------------------------
def _arm(body, *, ref, remark, boot="", scratch=(), note="", blk_cap=8):
    return {
        "body": body,
        "ref": ref,
        "remark": remark,
        "boot": boot,
        "scratch": tuple(scratch),
        "note": note,
        # Largest DEST window this arm may use. 7, not 8, for every arm containing a DestReuseBinary
        # (see the DEST-reuse window rule above) — a measured hardware/helper constraint, not a tune.
        "blk_cap": blk_cap,
    }


_FULL = _REMARK_GATE + _REMARK_UP + _DRAIN_OUT
_A_ONLY = _REMARK_GATE + _DRAIN_OUT

ARMS = {
    # ---- attribution ----
    "overhead": _arm(_OVERHEAD, ref=None, remark=_FULL, note="per-iteration CB scaffolding only"),
    "plain_add_x8": _arm(
        _PLAIN_ADD_X8,
        ref="add",
        remark=_DRAIN_OUT,
        note="the root's 8 plain blocked 48-tile reduce adds (held inputs, output drained between calls)",
    ),
    # ---- stage (a) alone: the m_eff-call bias walk vs its replacements ----
    "a_only_baseline": _arm(
        _a_stage(_A_BASELINE, "h_local"),
        ref="gate_silu",
        remark=_A_ONLY,
        boot=_BOOT_PACKER_SILU,
        note="stage (a) as shipped: M_EFF helper entries, packer SiLU",
    ),
    "a_only_hoist_rows": _arm(
        _a_stage(_A_HOIST_ROWS, "h_local"),
        ref="gate_silu",
        remark=_A_ONLY,
        boot=_BOOT_PACKER_SILU,
        note="stage (a), raw LLK, setup hoisted, M_EFF x HN_PAD-tile windows, packer SiLU (round 1's arm)",
    ),
    "a_only_blk_packer": _arm(
        _a_stage(_A_BLK_PACKER, "h_local"),
        ref="gate_silu",
        remark=_A_ONLY,
        boot=_BOOT_PACKER_SILU,
        note="stage (a), raw LLK, ONE pass, ELTWISE_BLK windows, packer SiLU",
    ),
    "a_only_chain": _arm(
        _a_stage(_A_CHAIN, "h_local"),
        ref="gate_silu",
        remark=_A_ONLY,
        note="stage (a), ONE helper chain add->SiLU->pack, math-thread SiLU",
    ),
    "a_only_sigappx": _arm(
        _a_stage(_A_ADD_TO_GATE_SUM + _A_SIGMOID_MUL, "h_local").replace("@APPX@", "true"),
        ref="gate_silu",
        remark=_A_ONLY,
        scratch=("gate_sum",),
        note="stage (a) with the SFPLUT approx sigmoid — isolates the SFPU spelling's whole cost",
    ),
    # ---- whole epilogue ----
    "baseline": _arm(
        _a_stage(_A_BASELINE, "gate_silu") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        boot=_BOOT_PACKER_SILU,
        scratch=("gate_silu", "up_sum"),
        note="SHIPPED: M_EFF+2 helper entries, 3 passes",
    ),
    "hoist_rows": _arm(
        _a_stage(_A_HOIST_ROWS, "gate_silu") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        boot=_BOOT_PACKER_SILU,
        scratch=("gate_silu", "up_sum"),
        note="round 1's non-graduated hoist: (a) raw+hoisted, (b)(c) as shipped",
    ),
    "blk_packer": _arm(
        _a_stage(_A_BLK_PACKER, "gate_silu") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        boot=_BOOT_PACKER_SILU,
        scratch=("gate_silu", "up_sum"),
        note="(a) ONE raw pass at ELTWISE_BLK width, packer SiLU; (b)(c) as shipped",
    ),
    "add_silu_chain": _arm(
        _a_stage(_A_CHAIN, "gate_silu") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum"),
        note="(a) ONE helper chain, math SiLU; (b)(c) as shipped -> 3 helper entries",
    ),
    "add_silu_chain_nr": _arm(
        _a_stage(_A_CHAIN_NR, "gate_silu") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum"),
        note="add_silu_chain with the wasted bfp8->bfp8 reconfig suppressed in (a)",
    ),
    "add_then_silu": _arm(
        _a_stage(_A_ADD_THEN_SILU, "gate_silu").replace("@P2SHAPE@", "blk_shape") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum", "gate_sum"),
        note="(a) = ONE blocked add + a SEPARATE blocked SFPU SiLU pass (the literal option 3)",
    ),
    "add_then_silu_dr": _arm(
        _a_stage(_A_ADD_THEN_SILU, "gate_silu").replace("@P2SHAPE@", "dr_shape") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum", "gate_sum"),
        note="add_then_silu with pass 2 at the DEST-reuse window (7) — the exact A/B twin of the sigmoid arms",
    ),
    "sigacc_mul": _arm(
        _a_stage(_A_ADD_TO_GATE_SUM + _A_SIGMOID_MUL, "gate_silu").replace("@APPX@", "false") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum", "gate_sum"),
        note="(a) 2-pass, SFPU spelled sigmoid(ACCURATE) + dest-reuse FPU mul",
    ),
    "sigappx_mul": _arm(
        _a_stage(_A_ADD_TO_GATE_SUM + _A_SIGMOID_MUL, "gate_silu").replace("@APPX@", "true") + _B_UP_ADD + _C_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu", "up_sum", "gate_sum"),
        note="(a) 2-pass, SFPU spelled sigmoid(SFPLUT APPX, what math_approx_mode=True asks for) + mul",
    ),
    "sigappx_fused": _arm(
        _SIGAPPX_FUSED,
        ref="epilogue",
        remark=_FULL,
        scratch=("up_sum", "gate_sum"),
        note="3 passes, SFPLUT sigmoid, BOTH multiplies as dest-reuse FPU ops. cb_gate_silu DELETED",
    ),
    "fuse_silu_mul": _arm(
        _FUSE_SILU_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("up_sum",),
        note="2 passes: up add staged, then gate add -> SiLU -> dest-reuse mul -> h. cb_gate_silu DELETED",
    ),
    "fuse_silu_mul_pw": _arm(
        _FUSE_SILU_MUL_PW,
        ref="epilogue",
        remark=_FULL,
        scratch=("up_sum",),
        note="fuse_silu_mul with ONE chain call per DEST window — prices per-window setup vs the blk=7 haircut",
    ),
    "fuse_up_mul": _arm(
        _FUSE_UP_MUL,
        ref="epilogue",
        remark=_FULL,
        scratch=("gate_silu",),
        note="2 passes, other order: gate add -> SiLU -> L1, then up add -> dest-reuse mul -> h",
    ),
    "single_pass": _arm(
        _SINGLE_PASS,
        ref="epilogue",
        remark=_FULL,
        scratch=(),
        note="ONE pass, zero L1 intermediates: both adds in DEST (window clamps to 4), SFPU DEST-DEST mul",
    ),
    "single_pass_sigappx": _arm(
        _SINGLE_PASS_SIGAPPX,
        ref="epilogue",
        remark=_FULL,
        scratch=(),
        note="single_pass with SiLU spelled as the SFPLUT approx sigmoid + a second SFPU mul",
    ),
}

VARIANTS = tuple(ARMS)
# Scratch-CB L1 cost, bfp8_b tiles (1088 B each): the op sizes each at M_BLOCK * HN_PAD = 48 tiles.
SCRATCH_TILES_OP = 48


def _kernel_source(arm, m_eff, hn_pad, kernel_iters, blk):
    spec = ARMS[arm]
    src = _PREAMBLE + "\nvoid kernel_main() {\n"
    # The op boots hw startup once with the gate stage's operand/pack triple.
    src += "    compute_kernel_hw_startup(cb_gate_acc, cb_reduce_gate_in, cb_gate_silu);\n"
    if spec["boot"]:
        src += spec["boot"] + "\n"
    src += _BOOT_CBS
    src += "    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {\n"
    src += spec["body"]
    if spec["remark"]:
        src += "        if (iter + 1 < KERNEL_ITERS) {\n" + spec["remark"] + "        }\n"
    src += "    }\n}\n"
    subs = {
        "@CB_GATE_ACC@": CB_GATE_ACC,
        "@CB_UP_ACC@": CB_UP_ACC,
        "@CB_REDUCE_GATE_IN@": CB_REDUCE_GATE_IN,
        "@CB_REDUCE_UP_IN@": CB_REDUCE_UP_IN,
        "@CB_GATE_SILU@": CB_GATE_SILU,
        "@CB_UP_SUM@": CB_UP_SUM,
        "@CB_GATE_SUM@": CB_GATE_SUM,
        "@CB_H_LOCAL@": CB_H_LOCAL,
        "@M_EFF@": m_eff,
        "@HN_PAD@": hn_pad,
        "@ITERS@": kernel_iters,
        "@BLK@": blk,
    }
    for token, value in subs.items():
        src = src.replace(token, str(value))
    assert "@" not in src, f"unsubstituted token in {arm}: {[t for t in src.split() if '@' in t][:4]}"
    return src


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    """Whole `shape` as a single-core height shard (row-major orientation)."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_tiles):
    tile_size = ttnn.tile_size(ttnn.bfloat8_b)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat8_b, page_size=tile_size)
    return ttnn.CBDescriptor(total_size=tile_size * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def moe_fused_swiglu_compute_config():
    """The op's `default_compute_kernel_config()` — the FROZEN precision contract. Never a lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def create_program_descriptor(input_tensors, output_tensor, *, m_eff, hn_pad, arm, kernel_iters=1, blk=8):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {VARIANTS}, got {arm!r}")
    if len(input_tensors) != 4:
        raise ValueError("needs 4 input tensors: [gate_acc, up_acc, reduce_gate_in, reduce_up_in]")
    for t in list(input_tensors) + [output_tensor]:
        if t.dtype != ttnn.bfloat8_b or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError("this bench uses bfloat8_b TILE_LAYOUT tensors throughout (as the op does)")
    if m_eff < 1 or hn_pad < 1 or kernel_iters < 1:
        raise ValueError("m_eff, hn_pad, kernel_iters must be positive")
    if hn_pad > 8:
        raise ValueError(f"hn_pad={hn_pad} exceeds DEST_AUTO_LIMIT=8 at fp32_dest_acc_en=False / half-sync")

    block_tiles = m_eff * hn_pad
    compute = ttnn.KernelDescriptor(
        kernel_source=_kernel_source(arm, m_eff, hn_pad, kernel_iters, blk),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[],
        config=moe_fused_swiglu_compute_config(),
    )
    gate_acc, up_acc, reduce_gate_in, reduce_up_in = input_tensors
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_GATE_ACC, gate_acc),
        ttnn.cb_descriptor_from_sharded_tensor(CB_UP_ACC, up_acc),
        ttnn.cb_descriptor_from_sharded_tensor(CB_REDUCE_GATE_IN, reduce_gate_in),
        ttnn.cb_descriptor_from_sharded_tensor(CB_REDUCE_UP_IN, reduce_up_in),
        _scratch_cb(CB_GATE_SILU, block_tiles),
        _scratch_cb(CB_UP_SUM, block_tiles),
        _scratch_cb(CB_GATE_SUM, block_tiles),
        ttnn.cb_descriptor_from_sharded_tensor(CB_H_LOCAL, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensors, *, m_eff, hn_pad, arm, kernel_iters=1, blk=8):
    m, n = m_eff * TILE, hn_pad * TILE
    device = input_tensors[0].device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m, n]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, create_sharded_memory_config((m, n))
    )
    descriptor = create_program_descriptor(
        input_tensors, output, m_eff=m_eff, hn_pad=hn_pad, arm=arm, kernel_iters=kernel_iters, blk=blk
    )
    return ttnn.generic_op([*input_tensors, output], descriptor)
