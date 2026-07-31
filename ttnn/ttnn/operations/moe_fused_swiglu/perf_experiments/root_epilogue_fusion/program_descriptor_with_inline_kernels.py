# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only isolated bake-off: moe_fused_swiglu's reduce-ROOT epilogue.

This is a `/perf-lab`-style micro-benchmark for ONE assigned idea — fuse the reduce root's
final-child epilogue into fewer / one compute pass — reconstructed from
`ttnn/ttnn/operations/moe_fused_swiglu/kernels/moe_fused_swiglu_compute.cpp` (the
`is_root && final_child` branch + the `compute_swiglu` zone). It does NOT touch the real op.

The math (identical in every variant, focus-shape tile counts m_eff=8, HN_PAD=6, 48 tiles):

    h = SiLU(gate_acc + last_gate_child) * (up_acc + last_up_child)

The production kernel computes this as THREE separate passes through L1, each paying its own
per-call setup:
  (a) 8 separate `add_bias_bcast_rows<Elementwise, ..., SiluActivation>` calls (one per token
      tile-row — the helper's bias index does not advance with in0_subblock,
      `bias_add_helpers.inl:141`), each its own reconfig + init + one 6-tile DEST window,
      SiLU riding the PACKER thread -> cb_gate_silu.
  (b) `add<input(cb_up_acc), input(cb_reduce_up_in), output(cb_up_acc)>(EltwiseShape::tiles(48))`
      — default WaitPolicy::PerTile/PopPolicy::PerTile inputs, so `eltwise_chain` clamps
      block_size to 1 (`chain_supports_block_v` requires Upfront/Cumulative/PerChunk — PerTile
      does not qualify): 48 SEPARATE one-tile DEST windows.
  (c) `mul<input(cb_gate_silu), input(cb_up_acc), output(cb_h_local)>(EltwiseShape::tiles(48))`
      — same per-tile clamp: another 48 separate one-tile DEST windows.
  Total: 8 + 48 + 48 = 104 DEST windows for one M-block's root epilogue.

Five variants, same math, same user precision contract (LoFi, approx SFPU, fp32_dest_acc_en=False,
dst_full_sync_en=False — DEST_AUTO_LIMIT=8 bf16 tiles, HN_PAD=6 fits one row per window):

  baseline         : verbatim reconstruction above (kernel_lib helpers, exactly as shipped).
  blocked_3pass    : same 3 CBs / 3 passes; (a) unchanged; (b)/(c) rewritten as raw per-row
                     (HN_PAD-tile) DEST windows instead of the default per-tile clamp.
                     Isolates "fix the trivial per-tile default" alone (8+8+8=24 windows).
  hoisted_bias     : blocked_3pass + (a) ALSO rewritten raw, with its reconfig/init hoisted ONCE
                     instead of repeated on 8 separate helper calls (24 windows, less MMIO).
  fused_up_mul     : hoisted_bias's (a), then (b)+(c) FUSED into one pass per row: FPU add
                     (up_acc + last_up_child) -> DEST, then `binary_dest_reuse_tiles` (DEST-reuse
                     multiply against cb_gate_silu) -> pack straight to cb_h_local. No up_sum CB
                     ever materializes. 8 (a) + 8 (fused b+c) = 16 windows.
  fused_math_silu  : identical to fused_up_mul, but (a)'s SiLU runs on the MATH thread (raw
                     `silu_tile`, in-DEST) instead of the PACKER thread — prices the loss of the
                     packer/math overlap `sfpu_activation_helpers.hpp:71-74` documents. Same
                     16-window count as fused_up_mul; isolates the packer-thread value alone.

DEST accounting (why there is no single-window 4-op fuse): HN_PAD=6 tiles already fills 6 of the
8 DEST_AUTO_LIMIT slots for ONE operand. A genuine single-window fuse of BOTH adds + SiLU + mul
needs the gate-sum AND the up-sum resident in DEST simultaneously — 12 tiles — which does not fit
under DEST_AUTO_LIMIT=8. `fused_up_mul`/`fused_math_silu` are the realizable 2-pass envelope of
that idea (avoids ever materializing up_sum in L1, at the cost of gate_silu still round-tripping
through L1 once — SiLU's packer-thread mechanism requires *some* L1 stop for the same reason).

Raw LLK justification (bypasses the "prefer helpers" rule, per this bench's isolated-bake-off
license): `add_bias_bcast_rows` / `add<>` / `mul<>` each pay a *per-call* reconfig_data_format_srca/
srcb + pack_reconfig_data_format + *_tiles_init, and the convenience wrappers clamp block_size to 1
for default (per-tile) input policies (`eltwise_chain.inl`'s `input_supports_block` requires
Upfront/Cumulative/PerChunk, not the default PerTile). Hand-rolling the same underlying LLK calls
(`add_tiles`/`add_tiles_init`, `binary_dest_reuse_tiles`/`binary_dest_reuse_tiles_init`, `pack_tile`,
`silu_tile`/`silu_tile_init`, `apply_activation_from_pack`) lets this bench hoist that setup ONCE
per pass and choose the DEST window width directly (HN_PAD tiles/row), which is the mechanism being
measured.

Implementation note — ONE kernel source PER VARIANT, no shared dispatch kernel. An earlier version
of this bench compiled all 5 methods into a single `kernel_main()` selected by an `if constexpr
(METHOD == ...)` ladder on a compile-time arg. That hung on device: `if constexpr` in a
non-template function does not prevent the *other* (unreached) branches from being compiled into
the same translation unit, so raw calls belonging to a different method (`binary_dest_reuse_tiles`,
`silu_tile`, `mul_tiles_init`, ...) were still present in the binary even for a run that never
executes them, and some JIT/LLK-side config (observed: the UNPACK<->MATH dest "context" handshake)
is apparently derived from what op *kinds* appear in the compiled kernel, not only which branch
executes at runtime — corrupting that handshake for the branch that DOES run. Confirmed by
bisection (`tt-probe.sh --dev`): a standalone single-method program never hung; the multi-method
dispatch kernel hung on its very first (`baseline`) call. Each variant below is therefore its own
independent kernel-source string with no dead branches, and `m_eff`/`hn_pad`/`kernel_iters` are
baked in as literals (an f-string) rather than threaded through as compile-time args.
"""

import ttnn

TILE = 32

# CB assignment.
CB_GATE_ACC = 0  # held input: gate accumulator before the last child (bfp8_b)
CB_UP_ACC = 1  # held input: up accumulator before the last child (bfp8_b)
CB_REDUCE_GATE_IN = 2  # held input: last child's gate partial (bfp8_b)
CB_REDUCE_UP_IN = 3  # held input: last child's up partial (bfp8_b)
CB_GATE_SILU = 4  # scratch: SiLU(gate_acc + reduce_gate_in)
CB_UP_SUM = 5  # scratch: up_acc + reduce_up_in (baseline / blocked_3pass / hoisted_bias only)
CB_H_LOCAL = 16  # output: SiLU(gate_sum) * up_sum

VARIANTS = ("baseline", "blocked_3pass", "hoisted_bias", "fused_up_mul", "fused_math_silu")
BASELINE = "baseline"

# =============================================================================
# Shared preamble: includes + CB id constants. Each variant appends its OWN kernel_main() body
# below this — no method lives in the same file as another (see the module-docstring note on why).
# =============================================================================
_PREAMBLE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

using namespace ckernel;
using namespace compute_kernel_lib;

constexpr uint32_t cb_gate_acc = {cb_gate_acc};
constexpr uint32_t cb_up_acc = {cb_up_acc};
constexpr uint32_t cb_reduce_gate_in = {cb_reduce_gate_in};
constexpr uint32_t cb_reduce_up_in = {cb_reduce_up_in};
constexpr uint32_t cb_gate_silu = {cb_gate_silu};
constexpr uint32_t cb_up_sum = {cb_up_sum};
constexpr uint32_t cb_h_local = {cb_h_local};

constexpr uint32_t M_EFF = {m_eff};
constexpr uint32_t HN_PAD = {hn_pad};
constexpr uint32_t KERNEL_ITERS = {kernel_iters};
constexpr uint32_t BLOCK_TILES = M_EFF * HN_PAD;

static_assert(HN_PAD <= compute_kernel_lib::DEST_AUTO_LIMIT, "one row must fit a DEST window");
"""

# Held-CB setup, identical for every variant: mark the tensor-backed operand CBs available once,
# wait once, never pop (raw absolute-index reads throughout the loop below). See the module
# docstring's note in the baseline/blocked_3pass re-mark block for the two exceptions that DO pop.
_HELD_CB_SETUP = r"""
    cb_reserve_back(cb_gate_acc, BLOCK_TILES);
    cb_push_back(cb_gate_acc, BLOCK_TILES);
    cb_reserve_back(cb_up_acc, BLOCK_TILES);
    cb_push_back(cb_up_acc, BLOCK_TILES);
    cb_reserve_back(cb_reduce_gate_in, BLOCK_TILES);
    cb_push_back(cb_reduce_gate_in, BLOCK_TILES);
    cb_reserve_back(cb_reduce_up_in, BLOCK_TILES);
    cb_push_back(cb_reduce_up_in, BLOCK_TILES);
    cb_wait_front(cb_gate_acc, BLOCK_TILES);
    cb_wait_front(cb_up_acc, BLOCK_TILES);
    cb_wait_front(cb_reduce_gate_in, BLOCK_TILES);
    cb_wait_front(cb_reduce_up_in, BLOCK_TILES);
"""

_BIAS_HELPER_CALL = r"""
        CircularBuffer gate_buf(cb_gate_acc), rg_buf(cb_reduce_gate_in), silu_buf(cb_gate_silu);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            add_bias_bcast_rows<
                BiasBroadcast::Elementwise,
                OutputCBLayout::SubblockMajor,
                bias_add_config::NoPostBias,
                SiluActivation>(
                gate_buf, rg_buf, silu_buf, BiasAddShape::of(1, 1, 1, HN_PAD), {}, m * HN_PAD);
        }
"""

# Re-mark whatever the real, popping `add_bias_bcast_rows` helper drained (baseline, blocked_3pass
# only — the raw-LLK stages in the other variants never pop their held operands).
_REMARK_GATE_ACC = r"""
            cb_reserve_back(cb_gate_acc, BLOCK_TILES);
            cb_push_back(cb_gate_acc, BLOCK_TILES);
            cb_wait_front(cb_gate_acc, BLOCK_TILES);
"""
_REMARK_UP_ACC_AND_REDUCE_UP = r"""
            cb_reserve_back(cb_up_acc, BLOCK_TILES);
            cb_push_back(cb_up_acc, BLOCK_TILES);
            cb_wait_front(cb_up_acc, BLOCK_TILES);
            cb_reserve_back(cb_reduce_up_in, BLOCK_TILES);
            cb_push_back(cb_reduce_up_in, BLOCK_TILES);
            cb_wait_front(cb_reduce_up_in, BLOCK_TILES);
"""
_DRAIN_H_LOCAL = r"""
            cb_wait_front(cb_h_local, BLOCK_TILES);
            cb_pop_front(cb_h_local, BLOCK_TILES);
"""


def _kernel_source(*, body, remark, m_eff, hn_pad, kernel_iters, boot_activation_init):
    preamble = _PREAMBLE.format(
        cb_gate_acc=CB_GATE_ACC,
        cb_up_acc=CB_UP_ACC,
        cb_reduce_gate_in=CB_REDUCE_GATE_IN,
        cb_reduce_up_in=CB_REDUCE_UP_IN,
        cb_gate_silu=CB_GATE_SILU,
        cb_up_sum=CB_UP_SUM,
        cb_h_local=CB_H_LOCAL,
        m_eff=m_eff,
        hn_pad=hn_pad,
        kernel_iters=kernel_iters,
    )
    return f"""{preamble}
void kernel_main() {{
    compute_kernel_hw_startup(cb_gate_acc, cb_reduce_gate_in, cb_gate_silu);
{boot_activation_init}
{_HELD_CB_SETUP}
    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {{
{body}
        if (iter + 1 < KERNEL_ITERS) {{
{remark}
        }}
    }}
}}
"""


_BOOT_PACKER_SILU = "    // Packer-thread SiLU boot init (sfpu_activation_helpers.hpp:71-74 — overlaps MATH).\n    ActivationInitHelper<KernelActivation::SILU>::init();"
_BOOT_MATH_SILU = (
    "    // MATH-thread SiLU boot init (prices dropping the packer-thread overlap).\n    silu_tile_init();"
)


def _baseline_source(m_eff, hn_pad, kernel_iters):
    body = f"""        {{
{_BIAS_HELPER_CALL}
        }}
        add<input(cb_up_acc), input(cb_reduce_up_in), output(cb_up_sum)>(EltwiseShape::tiles(BLOCK_TILES));
        mul<input(cb_gate_silu), input(cb_up_sum), output(cb_h_local)>(EltwiseShape::tiles(BLOCK_TILES));
"""
    remark = _REMARK_GATE_ACC + _REMARK_UP_ACC_AND_REDUCE_UP + _DRAIN_H_LOCAL
    return _kernel_source(
        body=body,
        remark=remark,
        m_eff=m_eff,
        hn_pad=hn_pad,
        kernel_iters=kernel_iters,
        boot_activation_init=_BOOT_PACKER_SILU,
    )


def _blocked_3pass_source(m_eff, hn_pad, kernel_iters):
    body = f"""        // (a) UNCHANGED from baseline -- isolates blocking (b)/(c) alone.
        {{
{_BIAS_HELPER_CALL}
        }}
        // (b) blocked, hoisted-init add: up_acc + reduce_up_in -> cb_up_sum, HN_PAD-tile windows.
        cb_reserve_back(cb_up_sum, BLOCK_TILES);
        {{
            reconfig_data_format_srca(cb_up_acc);
            reconfig_data_format_srcb(cb_reduce_up_in);
            pack_reconfig_data_format(cb_up_sum);
            add_tiles_init(cb_up_acc, cb_reduce_up_in);
            for (uint32_t m = 0; m < M_EFF; ++m) {{
                tile_regs_acquire();
                for (uint32_t c = 0; c < HN_PAD; ++c) {{
                    add_tiles(cb_up_acc, cb_reduce_up_in, m * HN_PAD + c, m * HN_PAD + c, c);
                }}
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < HN_PAD; ++c) {{
                    pack_tile(c, cb_up_sum);
                }}
                tile_regs_release();
            }}
        }}
        cb_push_back(cb_up_sum, BLOCK_TILES);

        // (c) blocked, hoisted-init mul: gate_silu * up_sum -> cb_h_local, HN_PAD-tile windows.
        cb_wait_front(cb_gate_silu, BLOCK_TILES);
        cb_wait_front(cb_up_sum, BLOCK_TILES);
        reconfig_data_format_srca(cb_gate_silu);
        reconfig_data_format_srcb(cb_up_sum);
        pack_reconfig_data_format(cb_h_local);
        mul_tiles_init(cb_gate_silu, cb_up_sum);
        cb_reserve_back(cb_h_local, BLOCK_TILES);
        for (uint32_t m = 0; m < M_EFF; ++m) {{
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {{
                mul_tiles(cb_gate_silu, cb_up_sum, m * HN_PAD + c, m * HN_PAD + c, c);
            }}
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < HN_PAD; ++c) {{
                pack_tile(c, cb_h_local);
            }}
            tile_regs_release();
        }}
        cb_push_back(cb_h_local, BLOCK_TILES);
        cb_pop_front(cb_gate_silu, BLOCK_TILES);
        cb_pop_front(cb_up_sum, BLOCK_TILES);
"""
    remark = _REMARK_GATE_ACC + _DRAIN_H_LOCAL
    return _kernel_source(
        body=body,
        remark=remark,
        m_eff=m_eff,
        hn_pad=hn_pad,
        kernel_iters=kernel_iters,
        boot_activation_init=_BOOT_PACKER_SILU,
    )


def _raw_bias_body():
    """(a) raw hoisted bias+SiLU, packer-thread activation, reconfig/init issued ONCE."""
    return """        reconfig_data_format_srca(cb_gate_acc);
        reconfig_data_format_srcb(cb_reduce_gate_in);
        pack_reconfig_data_format(cb_gate_silu);
        add_tiles_init(cb_gate_acc, cb_reduce_gate_in);
        cb_reserve_back(cb_gate_silu, BLOCK_TILES);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                add_tiles(cb_gate_acc, cb_reduce_gate_in, m * HN_PAD + c, m * HN_PAD + c, c);
            }
            tile_regs_commit();
            // Packer-thread SiLU replaces tile_regs_wait() (sfpu_activation_helpers.hpp).
            apply_activation_from_pack<KernelActivation::SILU>(HN_PAD);
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                pack_tile(c, cb_gate_silu);
            }
            tile_regs_release();
        }
        cb_push_back(cb_gate_silu, BLOCK_TILES);
"""


def _hoisted_bias_source(m_eff, hn_pad, kernel_iters):
    body = f"""        // (a) raw hoisted bias+SiLU (packer thread), reconfig/init ONCE instead of per row.
{_raw_bias_body()}
        // (b) blocked, hoisted-init add: up_acc + reduce_up_in -> cb_up_sum.
        cb_reserve_back(cb_up_sum, BLOCK_TILES);
        {{
            reconfig_data_format_srca(cb_up_acc);
            reconfig_data_format_srcb(cb_reduce_up_in);
            pack_reconfig_data_format(cb_up_sum);
            add_tiles_init(cb_up_acc, cb_reduce_up_in);
            for (uint32_t m = 0; m < M_EFF; ++m) {{
                tile_regs_acquire();
                for (uint32_t c = 0; c < HN_PAD; ++c) {{
                    add_tiles(cb_up_acc, cb_reduce_up_in, m * HN_PAD + c, m * HN_PAD + c, c);
                }}
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < HN_PAD; ++c) {{
                    pack_tile(c, cb_up_sum);
                }}
                tile_regs_release();
            }}
        }}
        cb_push_back(cb_up_sum, BLOCK_TILES);

        // (c) blocked, hoisted-init mul: gate_silu * up_sum -> cb_h_local.
        cb_wait_front(cb_gate_silu, BLOCK_TILES);
        cb_wait_front(cb_up_sum, BLOCK_TILES);
        reconfig_data_format_srca(cb_gate_silu);
        reconfig_data_format_srcb(cb_up_sum);
        pack_reconfig_data_format(cb_h_local);
        mul_tiles_init(cb_gate_silu, cb_up_sum);
        cb_reserve_back(cb_h_local, BLOCK_TILES);
        for (uint32_t m = 0; m < M_EFF; ++m) {{
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {{
                mul_tiles(cb_gate_silu, cb_up_sum, m * HN_PAD + c, m * HN_PAD + c, c);
            }}
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < HN_PAD; ++c) {{
                pack_tile(c, cb_h_local);
            }}
            tile_regs_release();
        }}
        cb_push_back(cb_h_local, BLOCK_TILES);
        cb_pop_front(cb_gate_silu, BLOCK_TILES);
        cb_pop_front(cb_up_sum, BLOCK_TILES);
"""
    remark = _DRAIN_H_LOCAL  # raw-LLK bias pass never pops its held operands
    return _kernel_source(
        body=body,
        remark=remark,
        m_eff=m_eff,
        hn_pad=hn_pad,
        kernel_iters=kernel_iters,
        boot_activation_init=_BOOT_PACKER_SILU,
    )


def _fused_source(m_eff, hn_pad, kernel_iters, *, math_silu):
    if math_silu:
        bias_body = """        reconfig_data_format_srca(cb_gate_acc);
        reconfig_data_format_srcb(cb_reduce_gate_in);
        pack_reconfig_data_format(cb_gate_silu);
        add_tiles_init(cb_gate_acc, cb_reduce_gate_in);
        cb_reserve_back(cb_gate_silu, BLOCK_TILES);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                add_tiles(cb_gate_acc, cb_reduce_gate_in, m * HN_PAD + c, m * HN_PAD + c, c);
            }
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                silu_tile(c);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                pack_tile(c, cb_gate_silu);
            }
            tile_regs_release();
        }
        cb_push_back(cb_gate_silu, BLOCK_TILES);
"""
        boot = _BOOT_MATH_SILU
    else:
        bias_body = _raw_bias_body()
        boot = _BOOT_PACKER_SILU

    fused_bc = """        // (b)+(c) FUSED: FPU add -> DEST, then a DEST-reuse multiply against cb_gate_silu,
        // packed straight to cb_h_local. cb_up_sum never materializes.
        //
        // NOTE (found via tt-probe --dev, LLK assert are_unpackers_AB_configured_correctly):
        // add_tiles needs the dual-operand AB unpack config (add_tiles_init); binary_dest_reuse_tiles
        // needs the single-operand A unpack config (binary_dest_reuse_tiles_init) -- DIFFERENT
        // unpacker configs. Hoisting both inits ONCE above the row loop (call once, alternate
        // ops many times) is exactly the bug: after row 0's dest-reuse calls leave the unpacker in
        // single-A mode, row 1's add_tiles silently runs under the wrong config. Both inits are
        // therefore re-issued every row (2 x M_EFF total), immediately before the op they configure --
        // still 8 fused windows instead of baseline's 96 (48+48), just with a per-row (not per-tile)
        // reinit cost, which is exactly the mechanism this candidate's price tag should carry.
        cb_wait_front(cb_gate_silu, BLOCK_TILES);
        pack_reconfig_data_format(cb_h_local);
        cb_reserve_back(cb_h_local, BLOCK_TILES);
        for (uint32_t m = 0; m < M_EFF; ++m) {
            reconfig_data_format_srca(cb_up_acc);
            reconfig_data_format_srcb(cb_reduce_up_in);
            add_tiles_init(cb_up_acc, cb_reduce_up_in);
            tile_regs_acquire();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                add_tiles(cb_up_acc, cb_reduce_up_in, m * HN_PAD + c, m * HN_PAD + c, c);
            }
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_gate_silu);
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_gate_silu, m * HN_PAD + c, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < HN_PAD; ++c) {
                pack_tile(c, cb_h_local);
            }
            tile_regs_release();
        }
        cb_push_back(cb_h_local, BLOCK_TILES);
        cb_pop_front(cb_gate_silu, BLOCK_TILES);
"""
    body = bias_body + fused_bc
    remark = _DRAIN_H_LOCAL
    return _kernel_source(
        body=body,
        remark=remark,
        m_eff=m_eff,
        hn_pad=hn_pad,
        kernel_iters=kernel_iters,
        boot_activation_init=boot,
    )


_SOURCE_BUILDERS = {
    "baseline": _baseline_source,
    "blocked_3pass": _blocked_3pass_source,
    "hoisted_bias": _hoisted_bias_source,
    "fused_up_mul": lambda m_eff, hn_pad, kernel_iters: _fused_source(m_eff, hn_pad, kernel_iters, math_silu=False),
    "fused_math_silu": lambda m_eff, hn_pad, kernel_iters: _fused_source(m_eff, hn_pad, kernel_iters, math_silu=True),
}


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================
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
    """The op's `default_compute_kernel_config()` — the fixed precision contract. Never a lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def create_program_descriptor(input_tensors, output_tensor, *, m_eff, hn_pad, variant, kernel_iters=1):
    if variant not in _SOURCE_BUILDERS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if len(input_tensors) != 4:
        raise ValueError("root_epilogue_fusion needs 4 input tensors: [gate_acc, up_acc, reduce_gate_in, reduce_up_in]")
    for t in list(input_tensors) + [output_tensor]:
        if t.dtype != ttnn.bfloat8_b or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError("root_epilogue_fusion uses bfloat8_b TILE_LAYOUT tensors throughout")
    if m_eff < 1 or hn_pad < 1 or kernel_iters < 1:
        raise ValueError("m_eff, hn_pad, kernel_iters must be positive")
    if hn_pad > 8:
        raise ValueError(f"hn_pad={hn_pad} exceeds DEST_AUTO_LIMIT=8 at fp32_dest_acc_en=False/half-sync")

    block_tiles = m_eff * hn_pad
    kernel_source = _SOURCE_BUILDERS[variant](m_eff, hn_pad, kernel_iters)

    compute = ttnn.KernelDescriptor(
        kernel_source=kernel_source,
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
        ttnn.cb_descriptor_from_sharded_tensor(CB_H_LOCAL, output_tensor),
    ]

    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensors, *, m_eff, hn_pad, variant, kernel_iters=1):
    """Allocate the h_local output and run one variant."""
    m, n = m_eff * TILE, hn_pad * TILE
    device = input_tensors[0].device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m, n]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config((m, n)),
    )
    descriptor = create_program_descriptor(
        input_tensors, output, m_eff=m_eff, hn_pad=hn_pad, variant=variant, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([*input_tensors, output], descriptor)
