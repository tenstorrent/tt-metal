# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — blocking selection + ProgramDescriptor.

Regimes built:
  R0     — DRAM interleaved: 2-D (m x n) split, DRAM window, per-core weight reads.
  R4/R5  — BLOCK / WIDTH sharded: the shard pins m (grid row) and the input-channel
           slice (grid column / WIDTH core). Each core gathers + tilizes the im2col of
           its own channel slice from its resident shard (+ halo over the NoC from the
           same-column shards) and multicasts it along its row (rotating sender); every
           core accumulates all S slices against its own output-column slice. Output is
           a shard-backed CB on the sharded output tensor (zero-copy).
  R3+R1  — HEIGHT sharded: the R4 path with one channel slice (S = 1, no K exchange). The
           halo-source "column" is every input shard in shard order over the grid; the output
           is HEIGHT-sharded over the first g_m input cores (core_m_tiles*32 rows each), and
           each weight K-block is read once from DRAM by core 0 and multicast to all of them.

Every knob (core split, bm / bk / bn, buffer depths) is defined once here and
every CB size, loop trip count and kernel arg derives from it.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

# Placement / L1-fit refusals are typed support refusals (a regime not built yet, e.g. R10 DRAM
# slicing), never a generic error: the harness reads them as deliberate refusals.
from ttnn.operations._op_contract import UnsupportedAxisValue

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
L1_NOC_ALIGN = 16

# ---- Module knobs (single source of truth) ----
CB_BUDGET_BYTES = 768 * 1024  # preferred per-core CB budget (perf lamp L5)
L1_HEADROOM_BYTES = 128 * 1024  # kept free under the hard cap (L1_SMALL, allocator slack)


def l1_cap_bytes():
    """Hard per-core CB cap: worker L1 unreserved size minus headroom."""
    return ttnn.get_max_worker_l1_unreserved_size() - L1_HEADROOM_BYTES


ACT_RM_DEPTH = 2  # cb_act_rm depth, in (1 tile-row x bk) quanta
WEIGHT_DEPTH = 2  # cb_weight_tiles depth, in (bk x bn) K-blocks
# R4/R5 cb_weight_tiles depth. The writer runs the weight stream depth - 2 K-blocks ahead of
# the act exchange (weight block j + lead is fetched / column-multicast around act block j), so
# the weight DRAM latency and column handshake leave the row exchange's critical chain. 2 = no lead.
SHARDED_WEIGHT_DEPTH = 6
OUT_RM_DEPTH = 2  # cb_out_rm depth, in (1 tile-row x bn) quanta
ACT_SEND_DEPTH = 1  # R4/R5 cb_act_send depth, in own (bm x bk) K-blocks (tilized, awaiting multicast);
# >= kblocks_per_slice tilizes the whole own slice at m-block start (the former own-slice prefetch).
ACT_RECV_DEPTH = 2  # R4/R5 cb_act_tiles depth, in received (bm x bk) K-blocks (double-buffered multicast)
# R4/R5: an own K-block is tilized (into cb_act_send) this many rounds before its own multicast round,
# instead of as soon as the previous own block was sent: compute then never waits on the reader's
# gather of a block that is not needed yet (that wait stalled the whole row's exchange). Clamped to
# [1, slices - 1]; slices - 1 = as soon as the send slot frees.
ACT_TILIZE_LEAD = 2
# R4/R5 (BLOCK) act exchange multicast channels. 2: rounds alternate between two channels (each with
# its own data-ready flag), so a receiver acks two rounds ahead and the owner of the next round sends
# it while the current one is still landing — two transfers in flight, the per-round handshake hidden
# behind the payload. Costs one more landing slot (cb_act_tiles is ACT_RECV_DEPTH + channels - 1 deep).
# 1 = one round in flight (ack the next round when the current one lands).
ACT_CHANNELS = 1
# dual_noc_act_mcast bench knobs (env, read per program build):
#   DNA_MODE 0 = HEAD helper act channel; 1 = raw act protocol from a DM_DYNAMIC_NOC writer,
#   DNA_B16/16 of each round's bytes on NOC0 (rest NOC1), DNA_CHUNK bytes per interleaved burst.
import os as _os

# R4/R5 cb_act_rm depth in whole own K-blocks (bm tile-rows x bk): the reader gathers the next own
# K-block while the row multicasts the other S-1 slices' blocks (K order interleaves the slices),
# so the gather leaves the compute critical path. Never below ACT_RM_DEPTH tile-rows.
ACT_RM_KBLOCKS = 1
# R4/R5 preferred CB footprint target (perf lamp L5). The shard-backed input / output CBs are
# extra (they are the tensors), so this sits well under CB_BUDGET_BYTES: SDXL L1 3x3 BLOCK must
# stay under production's 618,992 B per-core L1 peak including both shards and L1_SMALL.
SHARDED_CB_BUDGET_BYTES = 480 * 1024
MCAST_MAX_RECTANGLES = 3  # receiver rectangles one Mcast channel supports (mcast_protocol.hpp)
WEIGHT_MCAST = True  # R4: read each weight K-block once per grid column and multicast it down the column
# R4: the column's weight K-blocks are read + multicast by a rotating row (block w by row w % rows)
# instead of always grid row 0, so the DRAM read issue / barrier / multicast of every block is not
# serialized into one row's act-exchange loop.
WEIGHT_ROTATING_SENDER = True
# ... only while a weight K-block is small next to the act K-block it rides with (weight bytes / act
# bytes per K-block <= this): the moving multicast source then costs nothing. Heavier weight streams
# (ResNet layer4: 18 bf8b tiles per 27 act tiles) measured 81-95k cycles run to run rotating vs a
# steady 84k with the fixed row-0 sender.
WEIGHT_ROTATE_MAX_RATIO = 0.25
# R4 (BLOCK): store the prepared weight slice-major (weight_slice_tiles), so each grid column's weight
# K-blocks are contiguous pages over all DRAM banks.
WEIGHT_SLICE_MAJOR = True
BK_FLOOR = 4  # bk >= min(K_t, BK_FLOOR) on the first walk (perf lamp L7)
# R11 depthwise: cb_act_tiles depth in tilized tile-rows (taps x bn tiles each). Compute tilizes a
# tile-row and consumes it before tilizing the next, so one is enough; cb_act_rm (ACT_RM_DEPTH
# tile-rows) is what overlaps the reader's gather with compute.
DW_ACT_TILE_ROWS = 1
# R11 depthwise: cb_weight_tiles depth in (taps x bn) weight blocks — 2 lets the writer fetch the
# next (m, n)-block's tap weights while compute works on the current one.
DW_WEIGHT_DEPTH = 2
# R11 depthwise interleaved: minimum output tile-rows per core on the m split. Each core's DRAM
# window re-reads the kH-1 halo rows, and a depthwise window is one small DRAM page per stick, so
# the op is DRAM-request bound: taller per-core blocks cut the total request count.
DW_CORE_M_TILES_MIN = 1
# Tap-major im2col K-blocks (stride 1, dilation 1, whole-tile taps): the reader lays each K-block
# out as one contiguous 32-pixel x tap block per tap and compute tilizes one tap wide, so a tap's
# pixels along an output row are one NoC read (not one read per pixel per kernel-row band).
# bk becomes a whole number of taps. Enabled on the sharded placements in TAP_MAJOR_LAYOUTS: HEIGHT
# (R3, Refinement 6) and BLOCK / WIDTH (R4/R5, Refinement 9: a one-channel-tile slice's band is only
# 3 taps wide, so the per-pixel band reads were the gather's whole cost).
TAP_MAJOR_GATHER = True
TAP_MAJOR_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)
# R3 (HEIGHT): keep the whole weight slice resident in cb_weight_tiles (read + multicast once,
# before the first m-block) when it fits, instead of streaming SHARDED_WEIGHT_DEPTH K-blocks per
# m-block. Every HEIGHT core needs all of it, and the per-K-block read -> multicast rounds are
# strictly serial on the sender (no act exchange to hide them behind).
WEIGHTS_RESIDENT = True
# R3 with resident weights: one K-block per m-block + a whole-m-block-deep cb_act_rm, largest bm
# that fits SHARDED_CB_BUDGET_BYTES (gather of m-block i+1 overlaps compute of i). False / no fit:
# the K-blocked whole-shard walk below.
HEIGHT_SINGLE_K_PIPELINE = True
# Resident weights are read + multicast in this many chunks (K-row ranges): chunk c is multicast
# while the reads of chunks c+1.. are in flight. 1 = one read barrier, then one multicast.
WEIGHT_MCAST_CHUNKS = 1
# Single-K pipeline with resident weights: split K into (up to) this many K-blocks, each one weight
# multicast chunk, so the first m-block's matmul overlaps the weight multicast. 1 = one K-block.
HEIGHT_RESIDENT_KBLOCKS = 1
# Single-K pipeline: at least this many m-blocks per core (when the shard has the tile-rows), so
# the gather of m-block i+1 overlaps the compute of m-block i even when one block would fit.
HEIGHT_MIN_M_BLOCKS = 2
# R3 split gather: the writer RISC (idle once the resident weights are multicast) fetches + gathers
# the odd m-blocks into its own window / rm CBs while the reader takes the even ones; compute
# tilizes m-block i from the CB of RISC i % 2. Used with the single-K pipeline when it fits.
SPLIT_GATHER = False
# R3: place output shard r on the input core whose shard best covers its window (monotone in r),
# instead of on input core r. Identical for stride-1 convs; for strided convs it turns most of the
# window from far-away halo into the resident shard. The weight multicast then spans every input
# core (the ones holding no output shard receive only).
HEIGHT_ALIGN_OUTPUT = True
# R9 (technique 3): a 1x1 stride-1 TILE conv whose output tile-rows are its own shard's tile-rows uses
# the shard's tiles as the matmul in0 directly (HEIGHT) / as the multicast source (BLOCK / WIDTH): no
# untilize, gather or tilize. False = the general TILE path (untilize -> gather -> tilize).
TILE_DIRECT_IN0 = True
# Compute config policy, per (activation dtype, weight dtype): the single source for the
# descriptor's ComputeConfigDescriptor, the DEST capacity used by the subblock rule and the
# accumulator (partials) page format. The public signature takes no compute_kernel_config,
# so the op owns this policy (Refinement 2; measured PCC in changelog.md).
COMPUTE_POLICY = {
    # dtype: (math_fidelity, fp32_dest_acc_en) — weight dtype keyed below when it matters.
    ttnn.float32: (ttnn.MathFidelity.HiFi4, True),
    ttnn.bfloat16: (ttnn.MathFidelity.HiFi2, True),
    # bf8b activations (TILE only) untilize to bf16 before the gather: the bf16 policy.
    ttnn.bfloat8_b: (ttnn.MathFidelity.HiFi2, True),
}
# (activation dtype, weight dtype) overrides of COMPUTE_POLICY. bf16 x bf8b runs at LoFi: the bf8b
# weight mantissa (7 bits) is what LoFi's single fidelity phase consumes, it is the fidelity the
# ResNet50 reference runs at, and fp32 DEST still accumulates K exactly (Refinement 6 changelog).
COMPUTE_POLICY_BY_WEIGHT = {
    (ttnn.bfloat16, ttnn.bfloat8_b): (ttnn.MathFidelity.LoFi, True),
    (ttnn.bfloat8_b, ttnn.bfloat8_b): (ttnn.MathFidelity.LoFi, True),
}


def compute_policy(dtype, weight_dtype):
    """(math_fidelity, fp32_dest_acc_en, dst_tiles, acc_dtype) for a dtype pair."""
    fidelity, fp32_acc = COMPUTE_POLICY_BY_WEIGHT.get((dtype, weight_dtype), COMPUTE_POLICY[dtype])
    dst_tiles = 4 if fp32_acc else 8  # half-sync DEST capacity in tiles
    acc_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16  # partials page follows DEST width
    return fidelity, fp32_acc, dst_tiles, acc_dtype


# ---- CB indices (semantic names) ----
CB_ACT_WINDOW = 0
CB_ACT_RM = 1
CB_WEIGHT_TILES = 2
CB_BIAS_TILES = 3
CB_ACT_TILES = 4
CB_PARTIALS = 5
CB_BIASED = 6
CB_OUT_RM = 7  # R0: untilized tile-rows for the DRAM writer; R4/R5: the ragged-tail staging tile-row
CB_IN_SHARD = 8  # R4/R5: zero-copy view of the resident input shard
CB_OUT_SHARD = 9  # R4/R5: zero-copy view of the output shard (untilize target)
CB_ACT_SEND = 10  # R4/R5: own-slice tilized K-blocks awaiting the row multicast
CB_ACT_WINDOW2 = 11  # R3 split gather: the writer RISC's halo window (odd m-blocks)
CB_ACT_RM2 = 12  # R3 split gather: the writer RISC's RM im2col (odd m-blocks)
CB_WIN_TILES = 13  # TILE input: the window's tile-rows (DRAM) / halo tile-rows (sharded) awaiting untilize
CB_STICK_STAGE = 14  # R0 unaligned C_in: DRAM-aligned stick staging (two halves), copied into the window
# R0 unaligned C_in: sticks per staging half (one DRAM read barrier per half; the other half's
# window copies overlap it). Clamped to the window width.
STAGE_ROW_STICKS = 64

_ML = ttnn.TensorMemoryLayout
K_EXCHANGE_LAYOUTS = (_ML.BLOCK_SHARDED, _ML.WIDTH_SHARDED)


def _div_up(a, b):
    return -(-a // b)


def tile_bytes(dtype):
    """Bytes of one 32x32 tile of `dtype` (bf8b: 64 B exponents + 1024 B mantissas)."""
    return {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[dtype]


def rm_dtype(dtype):
    """Format of the row-major im2col pipeline (window, im2col rows, tilized in0): a bf8b TILE
    activation untilizes to bf16 (technique 1); other dtypes are carried as-is."""
    return ttnn.bfloat16 if dtype == ttnn.bfloat8_b else dtype


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _block_col_major(memory_config):
    """BLOCK with COL_MAJOR orientation: shard rows run along grid x, channel slices along y."""
    return (
        memory_config.memory_layout == _ML.BLOCK_SHARDED
        and memory_config.shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR
    )


def _block_core(start, s, r, col_major):
    """Logical core of BLOCK shard (channel slice s, shard row r)."""
    return ttnn.CoreCoord(start.x + r, start.y + s) if col_major else ttnn.CoreCoord(start.x + s, start.y + r)


def channel_slices(memory_config):
    """(S, slice_width) of a WIDTH/BLOCK-sharded input's channel split, else None.

    The single source prepare_conv2d_weights, validate() and the program agree on:
    S = shard columns (BLOCK: grid x for ROW_MAJOR orientation, grid y for COL_MAJOR — the
    transposed grid) or shard cores (WIDTH), slice_width = shard width in channels."""
    if memory_config.memory_layout not in K_EXCHANGE_LAYOUTS:
        return None
    spec = memory_config.shard_spec
    col_major = _block_col_major(memory_config)
    if memory_config.memory_layout == _ML.WIDTH_SHARDED and spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise UnsupportedAxisValue("conv2d_nhwc: WIDTH_SHARDED input needs ROW_MAJOR shard orientation")
    if memory_config.memory_layout == _ML.BLOCK_SHARDED:
        bb = spec.grid.bounding_box()
        S = (bb.end.y - bb.start.y + 1) if col_major else (bb.end.x - bb.start.x + 1)
    else:
        S = spec.grid.num_cores()
    return S, int(spec.shape[1])


def weight_slice_tiles(memory_config, C_in, C_out, groups):
    """BLOCK: the prepared weight's pages are stored slice-major — output-channel slice c's
    [K_t, nt_s] tile block is the contiguous page range [c*K_t*nt_s, (c+1)*K_t*nt_s), row-major —
    instead of the [K_t, N_t] row-major grid. A grid column's K-block is then one contiguous run of
    pages spread over every DRAM bank (row-major, N_t a multiple of the bank count puts a column's
    tiles in one or two banks: its weight stream was bank-bound). Returns nt_s, or 0 (row-major).
    The single source prepare_conv2d_weights and the writer agree on (the tensor shape is unchanged)."""
    if (
        not WEIGHT_SLICE_MAJOR
        or memory_config is None
        or memory_config.memory_layout != _ML.BLOCK_SHARDED
        or is_depthwise(C_in, groups)
        or C_out % TILE != 0
    ):
        return 0
    S, _ = channel_slices(memory_config)
    N_t = C_out // TILE
    return N_t // S if S > 1 and N_t % S == 0 else 0


def sharded_slices(memory_config, C_in):
    """(S, slice_width) the sharded program runs with: channel_slices for BLOCK / WIDTH, one
    whole-stick slice for HEIGHT (R3: every core holds full sticks; the prepared K order is the
    interleaved one, which is the S = 1 slice order). A HEIGHT shard may be wider than C_in
    (an unaligned C_in padded to the L1 granule): slice_width is the shard's stick width."""
    if memory_config.memory_layout != _ML.HEIGHT_SHARDED:
        return channel_slices(memory_config)
    spec = memory_config.shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise UnsupportedAxisValue("conv2d_nhwc: HEIGHT_SHARDED input needs ROW_MAJOR shard orientation")
    if int(spec.shape[1]) < C_in:
        raise UnsupportedAxisValue(f"conv2d_nhwc: HEIGHT shard width {int(spec.shape[1])} < C_in {C_in}")
    return 1, int(spec.shape[1])


def height_shard_cores(memory_config, num_shards):
    """Logical cores holding HEIGHT shards 0 .. num_shards-1 (ROW_MAJOR shard order)."""
    cores = sorted(ttnn.corerange_to_cores(memory_config.shard_spec.grid), key=lambda c: (c.y, c.x))
    return cores[:num_shards]


def is_depthwise(C_in, groups):
    """Regime R11: one input channel per group (C_out = m * C_in)."""
    return groups > 1 and groups == C_in


def native_groups(*, C_in, C_out, groups, elem_bytes, memory_layout):
    """Grouped conv run per group (R0, Refinement 8): every output-channel group is whole tiles and
    every input-channel group starts on a DRAM read boundary, so a core whose n range lies in one
    group reads only that group's C_in/g channels of each stick and its K is that group's
    kH x kW x C_in/g. The prepared weight is [K_g, C_out]: column c's rows are its own group's.
    Otherwise (sharded placements, groups that are not whole tiles) grouped convs run as a dense
    conv over a block-diagonal weight (zero outside the group): `groups`x the MACs."""
    if groups == 1 or is_depthwise(C_in, groups) or memory_layout != _ML.INTERLEAVED:
        return False
    C_in_g, C_out_g = C_in // groups, C_out // groups
    return C_out_g % TILE == 0 and C_in_g % TILE == 0 and (C_in_g * elem_bytes) % ttnn.get_dram_alignment() == 0


def conv_geometry(*, C_in, kH, kW, groups, elem_bytes, slices=None, native_group=False):
    """Prepared-K geometry shared by prepare_conv2d_weights and the kernels.

    Interleaved: K order [kh][kw x tap_width], each kh band padded to band_tiles tiles.
    Grouped: native_group -> tap_width is one group's channels (C_in/g); otherwise the dense
    (block-diagonal) geometry over all C_in.
    Channel-sliced (BLOCK/WIDTH, technique 6): K order [slice][kh][kw x slice_width],
    each (slice, kh) band padded to band_tiles tiles; tap_width = the shard's stick width."""
    granule = L1_NOC_ALIGN // elem_bytes
    if is_depthwise(C_in, groups):
        # R11: K = the kH*kW taps; one prepared tile-row per tap (row 0 = per-channel weights).
        # tap_width = the window / shard stick width in channels (interleaved: refined per core).
        S, tap_width = (1, C_in) if slices is None else slices
        if slices is not None and (tap_width % TILE != 0 or _div_up(C_in, tap_width) != S):
            raise UnsupportedAxisValue(
                f"conv2d_nhwc: depthwise needs whole-tile channel shards (shard width {tap_width}, C_in {C_in})"
            )
        return {
            "tap_width": tap_width,
            "band_tiles": _div_up(kW * tap_width, TILE),
            "K_t": kH * kW,
            "S": S,
            "depthwise": True,
        }
    if slices is None:
        C_in_g = C_in // groups if native_group else C_in
        tap_width = _div_up(C_in_g, granule) * granule
        S = 1
    else:
        S, tap_width = slices
        if tap_width % granule != 0:
            raise UnsupportedAxisValue(f"conv2d_nhwc: shard width {tap_width} is not L1-granule aligned")
        if _div_up(C_in, tap_width) != S:
            raise ValueError(f"conv2d_nhwc: {S} channel shards of width {tap_width} do not tile C_in={C_in}")
    band_tiles = _div_up(kW * tap_width, TILE)
    return {"tap_width": tap_width, "band_tiles": band_tiles, "K_t": S * kH * band_tiles, "S": S}


def _window_bounds(pix0, pix1, *, H, W, H_out, W_out, kH, kW, padding, stride, dilation):
    """Input window an m-block's pixels [pix0, pix1] read (regime R0 + R10 column clip).

    Returns (row_lo, row_hi, col_lo, last_hi, win_w, mid_lo, mid_hi) over flattened input
    rows (n*H + h). Window stick of input (row, col) = (row - row_lo) * win_w + (col - col_lo).
      * one output row  -> 2-D clip: every row holds columns [col_lo, last_hi], win_w = width.
      * several rows    -> contiguous flattened range, win_w = W: the first input row starts
        at col_lo and the last ends at last_hi (each only when that row is not padding-clamped,
        i.e. read by that one output row alone); middle rows are whole.
    Mirrors the reader kernel's window bound computation exactly."""
    hw_out = H_out * W_out
    n0, r0 = divmod(pix0, hw_out)
    n1, r1 = divmod(pix1, hw_out)
    ho0, wo0 = divmod(r0, W_out)
    ho1, wo1 = divmod(r1, W_out)
    raw_lo = ho0 * stride - padding
    raw_hi = ho1 * stride - padding + dilation * (kH - 1)
    row_lo = n0 * H + min(max(0, raw_lo), H - 1)
    row_hi = max(n1 * H + max(min(H - 1, raw_hi), 0), row_lo)
    c_first = min(max(0, wo0 * stride - padding), W - 1)
    c_last = max(min(W - 1, wo1 * stride - padding + dilation * (kW - 1)), 0)
    if n0 == n1 and ho0 == ho1:
        col_lo, last_hi = c_first, max(c_last, c_first)
        return row_lo, row_hi, col_lo, last_hi, last_hi - col_lo + 1, col_lo, last_hi
    col_lo = c_first if raw_lo >= 0 else 0
    last_hi = c_last if raw_hi <= H - 1 else W - 1
    return row_lo, row_hi, col_lo, last_hi, W, 0, W - 1


def _window_tile_rows(pix0, pix1, **kw):
    """TILE input: input tile-rows [lo, hi] (32 flattened sticks each) holding an m-block's window."""
    lo, hi = _window_stick_range(pix0, pix1, **kw)
    return lo // TILE, hi // TILE


def _window_sticks(pix0, pix1, **kw):
    row_lo, row_hi, col_lo, last_hi, win_w, _, _ = _window_bounds(pix0, pix1, **kw)
    hw_out = kw["H_out"] * kw["W_out"]
    if kw["dilation"] > 1 and pix0 // kw["W_out"] == pix1 // kw["W_out"] and pix0 // hw_out == pix1 // hw_out:
        # R0 dilated block inside one output row: only its kH kernel rows (the reader's dil_rows).
        return kw["kH"] * win_w
    return (row_hi - row_lo) * win_w + (last_hi - col_lo) + 1


def select_blocking(
    *,
    N,
    H,
    W,
    C_in,
    C_out,
    kH,
    kW,
    padding,
    stride,
    dilation,
    groups,
    elem_bytes,
    weight_tile_bytes,
    num_cores,
    has_bias,
    l1_cap_bytes,
    dtype,
    weight_dtype,
    activation="none",
    tile_in=False,
):
    _, _, dst_tiles, acc_dtype = compute_policy(dtype, weight_dtype)
    native_group = native_groups(
        C_in=C_in, C_out=C_out, groups=groups, elem_bytes=elem_bytes, memory_layout=_ML.INTERLEAVED
    )
    geo = conv_geometry(C_in=C_in, kH=kH, kW=kW, groups=groups, elem_bytes=elem_bytes, native_group=native_group)
    tap_width, band_tiles, K_t = geo["tap_width"], geo["band_tiles"], geo["K_t"]
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    M = N * H_out * W_out
    M_t = _div_up(M, TILE)
    N_t = _div_up(C_out, TILE)

    # Core split: m first, n only over the cores the m split leaves idle.
    core_m_tiles = _div_up(M_t, min(M_t, num_cores))
    if is_depthwise(C_in, groups):
        core_m_tiles = min(M_t, max(core_m_tiles, DW_CORE_M_TILES_MIN))
    g_m = _div_up(M_t, core_m_tiles)
    n_budget = max(1, num_cores // g_m)
    # Native groups: a core's n range lies in one group (core_n_tiles divides the group's tiles).
    group_n_tiles = (C_out // groups) // TILE if native_group else N_t
    core_n_tiles = next((d for d in _divisors(group_n_tiles) if N_t // d <= n_budget), group_n_tiles)
    g_n = N_t // core_n_tiles
    if g_m * g_n > num_cores:
        # Native groups with more groups than the m split leaves cores for: one group per core
        # column, the m split shrinks to fit.
        if g_n > num_cores:
            raise UnsupportedAxisValue(f"conv2d_nhwc: {g_n} channel groups exceed the {num_cores}-core grid")
        core_m_tiles = _div_up(M_t, num_cores // g_n)
        g_m = _div_up(M_t, core_m_tiles)

    T = TILE * TILE * elem_bytes  # act / out tile bytes
    T_w = weight_tile_bytes  # weight / bias tile bytes (weight_dtype)
    T_acc = tile_bytes(acc_dtype)  # accumulator page = DEST width
    win_kw = dict(H=H, W=W, H_out=H_out, W_out=W_out, kH=kH, kW=kW, padding=padding, stride=stride, dilation=dilation)
    dw = geo.get("depthwise", False)
    taps = kH * kW
    if dw:
        # R11: a core reads only its own channel tiles into the window (stick = core_n_tiles*32
        # channels at channel offset n_tile_start*32); K = taps x bn per n-block, one K-block.
        if C_out != C_in:
            raise UnsupportedAxisValue("conv2d_nhwc: depthwise channel multiplier (C_out != C_in) not built")
        tap_width = core_n_tiles * TILE

    def window_max(bm):
        best = 1
        for mi in range(g_m):
            m_start = mi * core_m_tiles
            m_tiles = min(core_m_tiles, M_t - m_start)
            for b in range(_div_up(m_tiles, bm)):
                t0 = m_start + b * bm
                t1 = min(t0 + bm, m_start + m_tiles)
                if tile_in:  # technique 2: whole input tile-rows, untilized per m-block
                    lo, hi = _window_tile_rows(t0 * TILE, min(t1 * TILE, M) - 1, **win_kw)
                    best = max(best, (hi - lo + 1) * TILE)
                else:
                    best = max(best, _window_sticks(t0 * TILE, min(t1 * TILE, M) - 1, **win_kw))
        return best

    T_in = tile_bytes(dtype)
    # TILE: the untilized window sticks are whole input tile columns (c_tiles x 32 channels).
    c_tiles = _div_up(tap_width, TILE)
    # R0 unaligned sticks: a window stick slot (tap_bytes) that is not a DRAM-read-aligned stride is
    # filled through the staging CB; a stick shorter than its slot leaves pad lanes zeroed once.
    stick_bytes = (tap_width if (dw or native_group) else C_in) * elem_bytes
    tap_bytes = tap_width * elem_bytes
    src_bytes = c_tiles * TILE * elem_bytes if tile_in else tap_bytes
    dram_align = ttnn.get_dram_alignment()
    staged = not tile_in and tap_bytes % dram_align != 0
    stage_stride = _div_up(stick_bytes, dram_align) * dram_align if staged else 0
    stage_row_sticks = min(STAGE_ROW_STICKS, W) if staged else 0
    stage_bytes = 2 * stage_row_sticks * stage_stride

    def footprint(bm, bk, bn, win):
        if dw:  # bk = taps * bn: one tilized tile-row, DW_WEIGHT_DEPTH tap-weight blocks
            return (
                win * tap_width * elem_bytes
                + ACT_RM_DEPTH * bk * T
                + DW_ACT_TILE_ROWS * bk * T
                + DW_WEIGHT_DEPTH * bk * T_w
                + bm * bn * T_acc
                + (core_n_tiles * T_w if has_bias else 0)
                + OUT_RM_DEPTH * bn * T
            )
        return (
            win * src_bytes
            + stage_bytes
            + (win // TILE * c_tiles * T_in if tile_in else 0)  # TILE: the window's tiles
            + ACT_RM_DEPTH * bk * T
            + bm * bk * T
            + WEIGHT_DEPTH * bk * bn * T_w
            + bm * bn * T_acc
            + (core_n_tiles * T_w if has_bias else 0)
            + OUT_RM_DEPTH * bn * T
        )

    k_divs_desc = sorted(_divisors(K_t), reverse=True)
    win_cache = {}

    def win_of(bm):
        if bm not in win_cache:
            win_cache[bm] = window_max(bm)
        return win_cache[bm]

    def walk(budget):
        if dw:
            # R11: the whole channel range per pass first (one contiguous read per kernel row),
            # then the whole m range; bk is pinned to taps x bn.
            for bn in sorted(_divisors(core_n_tiles), reverse=True):
                for bm in range(core_m_tiles, 0, -1):
                    win = win_of(bm)
                    if footprint(bm, taps * bn, bn, win) <= budget:
                        return bm, taps * bn, bn, win
            return None
        for bn in sorted(_divisors(core_n_tiles), reverse=True):
            # Step 2 (bn = core_n_tiles only): bm = core_m_tiles with the bk floor.
            if bn == core_n_tiles:
                bm = core_m_tiles
                win = win_of(bm)
                for bk in k_divs_desc:
                    if bk >= min(K_t, BK_FLOOR) and footprint(bm, bk, bn, win) <= budget:
                        return bm, bk, bn, win
            # Step 3: walk bm down, bk over all divisors.
            for bm in range(core_m_tiles, 0, -1):
                win = win_of(bm)
                bk = next((d for d in k_divs_desc if footprint(bm, d, bn, win) <= budget), None)
                if bk is not None:
                    return bm, bk, bn, win
        return None

    # The preferred budget first (perf lamp L5: footprint is a target); only a shape that
    # cannot block under it falls back to the hard per-core L1 cap.
    choice = walk(CB_BUDGET_BYTES) or walk(l1_cap_bytes)
    if choice is None:
        raise UnsupportedAxisValue("conv2d_nhwc: input window exceeds L1 — DRAM-slicing regime R10 not built")
    bm, bk, bn, window_sticks_max = choice

    sb_w = max(d for d in _divisors(bn) if d <= dst_tiles)
    sb_h = max(d for d in _divisors(bm) if d * sb_w <= dst_tiles)

    return {
        **geo,
        **_dw_plan_keys(dw, taps, bk, bn, tap_width, elem_bytes, C_in),
        "window_stick_bytes": stick_bytes,
        "native_group": native_group,
        "group_n_tiles": group_n_tiles if native_group else 0,
        "stage_stride": stage_stride,
        "stage_row_sticks": stage_row_sticks,
        "src_bytes": src_bytes,
        "window_zero_bytes": window_sticks_max * tap_bytes
        if (not staged and not tile_in and stick_bytes < tap_bytes)
        else 0,
        "tap_width": tap_width,  # depthwise: the core's own channel tiles
        "H_out": H_out,
        "W_out": W_out,
        "M": M,
        "M_t": M_t,
        "N_t": N_t,
        "N": N,
        "H": H,
        "W": W,
        "C_in": C_in,
        "C_out": C_out,
        "kH": kH,
        "kW": kW,
        "padding": padding,
        "stride": stride,
        "dilation": dilation,
        "elem_bytes": elem_bytes,
        "has_bias": has_bias,
        "dtype": dtype,
        "weight_dtype": weight_dtype,
        "activation": activation,
        "core_m_tiles": core_m_tiles,
        "g_m": g_m,
        "core_n_tiles": core_n_tiles,
        "g_n": g_n,
        "block_m_tiles": bm,
        "block_k_tiles": bk,
        "block_n_tiles": bn,
        "num_k_blocks": 1 if dw else K_t // bk,
        "num_n_blocks": core_n_tiles // bn,
        "act_rm_rows": ACT_RM_DEPTH,
        "weight_depth": DW_WEIGHT_DEPTH if dw else WEIGHT_DEPTH,
        "window_sticks_max": window_sticks_max,
        "out_subblock_h": sb_h,
        "out_subblock_w": sb_w,
        "T": T,
        "T_w": T_w,
        "T_acc": T_acc,
        "footprint": footprint(bm, bk, bn, window_sticks_max),
        "tile_in": tile_in,
        "win_c_tiles": c_tiles if tile_in else 0,
        "win_tile_rows": window_sticks_max // TILE if tile_in else 0,
        "halo_tile_rows": window_sticks_max // TILE if tile_in else 0,
    }


def _dw_plan_keys(dw, taps, bk, bn, tap_width, elem_bytes, C_in):
    """Plan keys that differ between the dense matmul and the R11 depthwise compute structure.

    weight block = weight_k_rows x bn tiles (dense: bk K-rows; depthwise: one tile per tap and
    channel tile). act_tile_rows = tilized tile-rows resident in cb_act_tiles (dense: the m-block,
    `None` = block_m_tiles). window_stick_bytes = one window stick (dense: the whole C_in stick;
    depthwise interleaved: the core's own channel tiles)."""
    return {
        "depthwise": dw,
        "taps": taps if dw else 0,
        "weight_k_rows": taps if dw else bk,
        "act_tile_rows": DW_ACT_TILE_ROWS if dw else None,
        "window_stick_bytes": (tap_width if dw else C_in) * elem_bytes,
    }


def _window_stick_range(pix0, pix1, **kw):
    """Flattened input-stick range [lo, hi] covering an m-block's window (superset of the
    2-D clip). Mirrors the reader's sharded window."""
    row_lo, row_hi, col_lo, last_hi, _, _, _ = _window_bounds(pix0, pix1, **kw)
    return row_lo * kw["W"] + col_lo, row_hi * kw["W"] + last_hi


def _halo_sticks(lo, hi, own_lo, own_hi):
    """Window sticks [lo, hi] not in the resident shard [own_lo, own_hi): before + after runs."""
    before = max(0, min(own_lo, hi + 1) - lo)
    after = max(0, hi + 1 - max(own_hi, lo))
    return before + after


def select_blocking_sharded(
    *,
    N,
    H,
    W,
    C_in,
    C_out,
    kH,
    kW,
    padding,
    stride,
    dilation,
    groups,
    elem_bytes,
    weight_tile_bytes,
    has_bias,
    memory_layout,
    slices,
    shard_h,
    rows_in,
    l1_cap_bytes,
    dtype,
    weight_dtype,
    activation="none",
    tile_in=False,
):
    """Regimes R4 (BLOCK) / R5 (WIDTH): the shard pins the core split.

    m: output tile-rows over the R_in grid rows (`core_m_tiles` each, whole-shard block by
    default, clamped by L1 to a divisor). n: output-channel tiles over the S slices,
    `ceil(N_t/S)` each (trailing tiles zero, technique 5); bn = the whole slice (the output
    shard row). k: own-slice K-blocks of bk tiles (a divisor of the slice's kH*band_tiles);
    the largest that fits — every K-block is one multicast handshake (technique 8)."""
    _, _, dst_tiles, acc_dtype = compute_policy(dtype, weight_dtype)
    S, cw = slices
    granule = L1_NOC_ALIGN // elem_bytes
    # HEIGHT: the K tap is the prepared (interleaved) one — C_in rounded to the L1 granule — while a
    # shard stick may be wider (src_bytes). BLOCK / WIDTH: the tap is the shard's slice width.
    is_h = memory_layout == _ML.HEIGHT_SHARDED
    geo_slices = (1, _div_up(C_in, granule) * granule) if (is_h and not is_depthwise(C_in, groups)) else slices
    geo = conv_geometry(C_in=C_in, kH=kH, kW=kW, groups=groups, elem_bytes=elem_bytes, slices=geo_slices)
    tap_width, band_tiles, K_t = geo["tap_width"], geo["band_tiles"], geo["K_t"]
    slice_k_tiles = kH * band_tiles
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    M = N * H_out * W_out
    M_t = _div_up(M, TILE)
    N_t = _div_up(C_out, TILE)
    NHW = N * H * W
    dw = geo.get("depthwise", False)
    taps = kH * kW
    if dw and C_out != C_in:
        raise UnsupportedAxisValue("conv2d_nhwc: depthwise channel multiplier (C_out != C_in) not built")
    # R11: a depthwise core's outputs are exactly its own channel slice (no K exchange).
    nt_s = cw // TILE if dw else _div_up(N_t, S)
    if memory_layout == _ML.WIDTH_SHARDED:
        core_m_tiles, g_m = M_t, 1
    else:
        core_m_tiles = _div_up(M_t, rows_in)
        g_m = _div_up(M_t, core_m_tiles)
    # WIDTH output shards are exactly M rows tall: a partial last tile-row is staged and
    # copied (writer) instead of untilized in place. BLOCK shards are allocated whole.
    # A TILE output shard is whole tile-rows: no tail.
    out_tail = memory_layout == _ML.WIDTH_SHARDED and M % TILE != 0 and not tile_in

    T = TILE * TILE * elem_bytes
    T_w = weight_tile_bytes
    T_acc = tile_bytes(acc_dtype)
    win_kw = dict(H=H, W=W, H_out=H_out, W_out=W_out, kH=kH, kW=kW, padding=padding, stride=stride, dilation=dilation)
    # R3: output shard r is computed on (and placed on) the core of input shard out_shards[r].
    # BLOCK: grid row r. HEIGHT: the input shard whose sticks best cover the shard's window, so the
    # halo is small and comes from neighbours (strided convs: output row r's window sits ~stride x
    # further into the input than shard r; HEIGHT_ALIGN_OUTPUT off = shard r, the first g_m cores).
    # Resident weights (R3, dense): the whole weight slice lives in L1 for the kernel's life.
    # Cleared below (streamed K-blocks, unaligned output) when no resident blocking fits.
    weights_resident = WEIGHTS_RESIDENT and memory_layout == _ML.HEIGHT_SHARDED and not dw
    out_shards = list(range(g_m))
    if weights_resident and HEIGHT_ALIGN_OUTPUT:
        out_shards = []
        for r in range(g_m):
            lo, hi = _window_stick_range(r * core_m_tiles * TILE, min((r + 1) * core_m_tiles * TILE, M) - 1, **win_kw)
            first = out_shards[-1] + 1 if out_shards else 0
            last = rows_in - (g_m - r)  # leave a core for every later output shard
            cands = range(max(first, min(lo // shard_h, last)), min(last, hi // shard_h) + 1)
            best = max(
                cands or [max(first, min(lo // shard_h, last))],
                key=lambda c: (min(hi + 1, (c + 1) * shard_h) - max(lo, c * shard_h), -c),
            )
            out_shards.append(best)

    def halo_max(bm):
        best = 0
        for r in range(g_m):
            c = out_shards[r]
            own_lo, own_hi = (
                (0, NHW) if memory_layout == _ML.WIDTH_SHARDED else (c * shard_h, min(NHW, (c + 1) * shard_h))
            )
            m_start = r * core_m_tiles
            m_tiles = min(core_m_tiles, M_t - m_start)
            for b in range(_div_up(m_tiles, bm)):
                t0 = m_start + b * bm
                t1 = min(t0 + bm, m_start + m_tiles)
                lo, hi = _window_stick_range(t0 * TILE, min(t1 * TILE, M) - 1, **win_kw)
                best = max(best, _halo_sticks(lo, hi, own_lo, own_hi))
        return best

    # S == 1: the core owns all of K — no exchange; compute tilizes straight into in0 (as R0).
    # Depthwise: every channel slice is self-contained — no exchange at any S.
    act_mcast = S > 1 and not dw

    # Two act channels (BLOCK only: a WIDTH group may hold sinks, which run one channel) need one more
    # landing slot: two rounds in flight while compute works on a third.
    act_channels = ACT_CHANNELS if (act_mcast and memory_layout == _ML.BLOCK_SHARDED and S > 1) else 1
    recv_depth = ACT_RECV_DEPTH + act_channels - 1

    def send_depth(bk):
        return min(ACT_SEND_DEPTH, slice_k_tiles // bk)

    def rm_rows(bm, deep=False):
        # deep (single-K pipeline): one whole m-block ahead is enough — compute pops m-block i's rows
        # at its tilize, so the reader fills them with m-block i+1 during the matmul.
        if act_mcast:
            return max(ACT_RM_DEPTH, ACT_RM_KBLOCKS * bm)
        return ACT_RM_KBLOCKS * bm if deep else ACT_RM_DEPTH

    # TILE input (technique 1): each core untilizes ONE contiguous range of input tile-rows, once
    # per kernel: the union of its whole m range's window and its own shard. Its own tile-rows come
    # from the resident shard (zero-copy); the rest (halo) are fetched as tiles from the owning shards.
    T_in = tile_bytes(dtype)
    # TILE: untilized window sticks are whole input tile columns; RM: the shard stick.
    c_tiles = _div_up(cw, TILE)
    src_bytes = c_tiles * TILE * elem_bytes if tile_in else cw * elem_bytes

    def own_range(r):
        if memory_layout == _ML.WIDTH_SHARDED:
            return 0, NHW
        c = out_shards[r]
        return c * shard_h, min(NHW, (c + 1) * shard_h)

    def tile_windows():
        wins = []
        for r in range(g_m):
            lo, hi = _window_stick_range(r * core_m_tiles * TILE, min((r + 1) * core_m_tiles * TILE, M) - 1, **win_kw)
            own_lo, own_hi = own_range(r)
            own_t0, own_t1 = own_lo // TILE, _div_up(own_hi, TILE)
            u0, u1 = min(lo // TILE, own_t0), max(hi // TILE + 1, own_t1)
            wins.append((u0, own_t0 - u0, own_t1 - own_t0, u1 - own_t1))  # (first, before, own, after)
        return wins

    def footprint(bm, bk, halo, deep=False, roles=1):
        if tile_in:
            wins = tile_windows()
            win_rows = max(b + o + a for _, b, o, a in wins)
            halo_rows = max(max(b + a for _, b, _, a in wins), 1)
            # The RM window replaces the per-block halo scratch; plus the halo tile staging.
            halo = 0
            extra = win_rows * TILE * src_bytes + halo_rows * c_tiles * T_in
        else:
            extra = 0
        return extra + _footprint(bm, bk, halo, deep, roles)

    def _footprint(bm, bk, halo, deep=False, roles=1):
        if dw:  # bk = taps * nt_s (bn = the whole slice: the output shard row)
            return (
                max(halo, 1) * tap_width * elem_bytes
                + ACT_RM_DEPTH * bk * T
                + DW_ACT_TILE_ROWS * bk * T
                + DW_WEIGHT_DEPTH * bk * T_w
                + bm * nt_s * T_acc
                + (nt_s * T_w if has_bias else 0)
                + (nt_s * T if out_tail else 0)
            )
        return (
            roles * (max(halo, 1) if not tile_in else 0) * src_bytes
            + roles * rm_rows(bm, deep) * bk * T
            + (send_depth(bk) * bm * bk * T if act_mcast else 0)
            + (recv_depth if act_mcast else 1) * bm * bk * T
            + (slice_k_tiles if weights_resident else SHARDED_WEIGHT_DEPTH * bk) * nt_s * T_w
            + bm * nt_s * T_acc
            + (nt_s * T_w if has_bias else 0)
            + (nt_s * T if out_tail else 0)
        )

    tap_tiles = tap_width // TILE
    tap_major = (
        TAP_MAJOR_GATHER
        and memory_layout in TAP_MAJOR_LAYOUTS
        and not dw
        and stride == 1
        and dilation == 1
        and tap_width % TILE == 0
        and src_bytes == tap_width * elem_bytes
    )

    def k_divs(tm):
        return (
            [taps * nt_s]
            if dw
            else sorted((d for d in _divisors(slice_k_tiles) if not tm or d % tap_tiles == 0), reverse=True)
        )

    k_divs_desc = k_divs(tap_major)

    def walk(budget):
        # Whole-shard bm first (shrinking bm re-streams the weights per m-block), then the
        # largest bk: every K-block is one multicast handshake + one partials pass (technique 8).
        for bm in sorted(_divisors(core_m_tiles), reverse=True):
            halo = halo_max(bm)
            bk = next((d for d in k_divs_desc if footprint(bm, d, halo) <= budget), None)
            if bk is not None:
                return bm, bk, halo
        return None

    def walk_single_k(budget, roles=1):
        # R3 with resident weights: extra m-blocks re-read nothing, so take the whole K as ONE
        # K-block (the matmul accumulates in DEST: no packer-L1 partials passes, one tilize ->
        # matmul switch per m-block) and make cb_act_rm one whole m-block deep, so the reader
        # gathers m-block i+1 while compute works on m-block i. Largest bm that fits.
        bk = slice_k_tiles // resident_kblocks
        bms = [d for d in _divisors(core_m_tiles) if core_m_tiles // d >= HEIGHT_MIN_M_BLOCKS or d == 1]
        for bm in sorted(bms, reverse=True):
            halo = halo_max(bm)
            if footprint(bm, bk, halo, True, roles) <= budget and (roles == 1 or _div_up(core_m_tiles, bm) > 1):
                return bm, bk, halo
        return None

    # Resident weights arrive as one multicast chunk per K-block: the first m-block's matmul starts
    # on K-block 0 while the later chunks are still in flight. The largest count <= the knob that
    # divides K into whole taps (tap-major).
    resident_kblocks = next(
        (
            d
            for d in range(HEIGHT_RESIDENT_KBLOCKS, 0, -1)
            if slice_k_tiles % d == 0 and (not tap_major or (slice_k_tiles // d) % tap_tiles == 0)
        ),
        1,
    )

    deep_rm = False
    choice = None
    roles = 1
    if weights_resident and HEIGHT_SINGLE_K_PIPELINE:
        if SPLIT_GATHER and not tile_in:  # TILE: one untilized window per core, gathered by the reader
            choice = walk_single_k(SHARDED_CB_BUDGET_BYTES, 2)
            roles = 2 if choice else 1
        choice = choice or walk_single_k(SHARDED_CB_BUDGET_BYTES)
        deep_rm = choice is not None
    if weights_resident and not choice:
        choice = walk(SHARDED_CB_BUDGET_BYTES)  # resident weights, K-blocked whole-shard walk
    if not choice:
        # Too big to keep resident: stream the weights per K-block; the output stays on the first
        # g_m cores (the streamed column multicast needs the compute cores as its group).
        weights_resident = False
        out_shards = list(range(g_m))

        def walk_all():
            return walk(SHARDED_CB_BUDGET_BYTES) or walk(CB_BUDGET_BYTES) or walk(l1_cap_bytes)

        choice = walk_all()
        if tap_major and act_mcast and tap_tiles > 1:
            # K exchange: a K-block of whole taps can be smaller than the band walk's (a 2-tile tap
            # rules out odd bk), and every extra K-block is one more exchange round — more than the
            # tap-major gather saves (SDXL 3x3 64-channel slices: bk 3 -> 2, +10%). Keep tap-major
            # only when its K-block is as large.
            k_divs_desc = k_divs(False)
            band = walk_all()
            if band is not None and (choice is None or band[0] * band[1] > choice[0] * choice[1]):
                tap_major, choice = False, band
            else:
                k_divs_desc = k_divs(True)
    if not choice:
        raise UnsupportedAxisValue("conv2d_nhwc: sharded K-exchange blocks do not fit L1")
    bm, bk, halo = choice
    bn = nt_s
    kbps_ = 1 if dw else slice_k_tiles // bk
    # R9: in0 = the shard's tiles. Needs whole-tile im2col == the shard (1x1, s1, p0), one K-block per
    # slice (a K-block = bm whole shard tile-rows x the slice's channel tiles) and output tile-row r
    # on the core holding input tile-row r (shard height = core_m_tiles tile-rows, shard r -> output r).
    direct_in0 = (
        TILE_DIRECT_IN0
        and tile_in
        and not dw
        and kH == 1
        and kW == 1
        and stride == 1
        and padding == 0
        and kbps_ == 1
        and bk * TILE == tap_width
        and core_m_tiles * TILE == shard_h
        and out_shards == list(range(g_m))
    )
    sb_w = max(d for d in _divisors(bn) if d <= dst_tiles)
    sb_h = max(d for d in _divisors(bm) if d * sb_w <= dst_tiles)
    kbps = 1 if dw else slice_k_tiles // bk
    return {
        **geo,
        **_dw_plan_keys(dw, taps, bk, bn, tap_width, elem_bytes, C_in),
        "k_exchange": True,
        "groups": groups,
        "act_mcast": act_mcast,
        "src_bytes": src_bytes,
        "stage_stride": 0,
        "stage_row_sticks": 0,
        "window_zero_bytes": 0,
        "native_group": False,
        "tap_major": tap_major,
        "tilize_w": tap_tiles if tap_major else bk,
        "act_send_depth": send_depth(bk),
        "act_tilize_lead": max(1, min(ACT_TILIZE_LEAD, S - 1)) if act_mcast else 0,
        "act_channels": act_channels,
        "act_recv_depth": recv_depth,
        "act_rm_rows": rm_rows(bm, deep_rm),
        # Resident: the CB holds every K-block (depth = the K-block count).
        "weight_depth": DW_WEIGHT_DEPTH if dw else (slice_k_tiles // bk if weights_resident else SHARDED_WEIGHT_DEPTH),
        "weights_resident": weights_resident,
        # Resident: one multicast chunk per K-block when K is blocked, else WEIGHT_MCAST_CHUNKS.
        "weight_chunk_rows": bk if kbps > 1 else _div_up(slice_k_tiles, WEIGHT_MCAST_CHUNKS),
        # Every grid row must run the same number of m-blocks (= column multicast rounds).
        # Depthwise weights are a few tap tiles per slice: each core reads its own.
        # HEIGHT (R1): every core needs the same weights (dense and depthwise): one group.
        "weight_mcast": WEIGHT_MCAST
        and (memory_layout == _ML.HEIGHT_SHARDED or (memory_layout == _ML.BLOCK_SHARDED and not dw))
        and g_m > 1
        # Resident weights are multicast once, before any m-block: no per-m-block rounds to match.
        and (
            weights_resident or len({_div_up(min(core_m_tiles, M_t - r * core_m_tiles), bm) for r in range(g_m)}) == 1
        ),
        "memory_layout": memory_layout,
        "slice_width": cw,
        "shard_h": shard_h,
        "rows_in": rows_in,
        "H_out": H_out,
        "W_out": W_out,
        "M": M,
        "M_t": M_t,
        "N_t": N_t,
        "N": N,
        "H": H,
        "W": W,
        "C_in": C_in,
        "C_out": C_out,
        "kH": kH,
        "kW": kW,
        "padding": padding,
        "stride": stride,
        "dilation": dilation,
        "elem_bytes": elem_bytes,
        "has_bias": has_bias,
        "dtype": dtype,
        "weight_dtype": weight_dtype,
        "activation": activation,
        "core_m_tiles": core_m_tiles,
        "g_m": g_m,
        "core_n_tiles": nt_s,
        "g_n": S,
        "block_m_tiles": bm,
        "block_k_tiles": bk,
        "block_n_tiles": bn,
        "kblocks_per_slice": kbps,
        "num_k_blocks": kbps if dw else S * kbps,
        "num_n_blocks": 1,
        "window_sticks_max": max(b + o + a for _, b, o, a in tile_windows()) * TILE if tile_in else max(halo, 1),
        "out_tail": out_tail,
        "out_subblock_h": sb_h,
        "out_subblock_w": sb_w,
        "T": T,
        "T_w": T_w,
        "T_acc": T_acc,
        "footprint": footprint(bm, bk, halo, deep_rm, roles),
        "num_gather_roles": roles,
        "out_shards": out_shards,
        "tile_in": tile_in,
        "direct_in0": direct_in0,
        "win_c_tiles": c_tiles if tile_in else 0,
        "tile_windows": tile_windows() if tile_in else None,
        "win_tile_rows": max(b + o + a for _, b, o, a in tile_windows()) if tile_in else 0,
        "halo_tile_rows": max(max(b + a for _, b, _, a in tile_windows()), 1) if tile_in else 0,
    }


def output_memory_config(plan, input_memory_config):
    """Sharded output placement (R4/R5): same TensorMemoryLayout as the input, one
    ceil(N_t/S)-tile-wide column slice per input-channel slice (the shards together may be
    wider than C_out, technique 5). BLOCK: core_m_tiles*32 rows per grid row; WIDTH: all M rows."""
    p = plan
    spec = input_memory_config.shard_spec
    width = p["core_n_tiles"] * TILE
    if p["memory_layout"] == _ML.BLOCK_SHARDED:
        start = spec.grid.bounding_box().start
        end = _block_core(start, p["g_n"] - 1, p["g_m"] - 1, _block_col_major(input_memory_config))
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(start, end)])
        shape = [p["core_m_tiles"] * TILE, width]
    elif p["memory_layout"] == _ML.HEIGHT_SHARDED:
        # R3: input cores out_shards[r], core_m_tiles whole tile-rows each (all C_out channels).
        in_cores = height_shard_cores(input_memory_config, p["rows_in"])
        grid = _core_range_set([in_cores[c] for c in p["out_shards"]])
        shape = [p["core_m_tiles"] * TILE, width]
    else:
        grid = spec.grid
        # A TILE output shard is whole tile-rows.
        shape = [p["M_t"] * TILE if p.get("tile_in") else p["M"], width]
    return ttnn.MemoryConfig(p["memory_layout"], ttnn.BufferType.L1, ttnn.ShardSpec(grid, shape, spec.orientation))


def _sharded_core_map(plan, input_memory_config):
    """[(logical core, slice s, row r)] of the compute cores, row-major (= the multicast
    groups' order: group = one grid row (BLOCK) / all WIDTH cores, phase = slice)."""
    p = plan
    spec = input_memory_config.shard_spec
    if p["memory_layout"] == _ML.BLOCK_SHARDED:
        start = spec.grid.bounding_box().start
        cm = _block_col_major(input_memory_config)
        return [(_block_core(start, s, r, cm), s, r) for r in range(p["g_m"]) for s in range(p["g_n"])]
    if p["memory_layout"] == _ML.HEIGHT_SHARDED:
        in_cores = height_shard_cores(input_memory_config, p["rows_in"])
        return [(in_cores[c], 0, r) for r, c in enumerate(p["out_shards"])]
    cores = sorted(ttnn.corerange_to_cores(spec.grid), key=lambda c: (c.y, c.x))
    return [(c, s, 0) for s, c in enumerate(cores[: p["g_n"]])]


def _cb(index, dtype, page_size, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _interleaved_core_map(plan, grid):
    """[(logical core, ni, mi)]: the first g_m*g_n cores of the grid, row-major."""
    num_active = plan["g_m"] * plan["g_n"]
    return [(ttnn.CoreCoord(c % grid.x, c // grid.x), c % plan["g_n"], c // plan["g_n"]) for c in range(num_active)]


def _core_range_set(coords):
    """Rectangles over a row-major core list, iterating in the list's row-major order: one range per
    contiguous x run of a grid row; consecutive rows merge only when each is one identical run."""
    rows = {}
    for c in coords:
        rows.setdefault(c.y, []).append(c.x)
    ranges = []  # [y0, (x0, x1), y1, single-run row]
    for y in sorted(rows):
        xs = sorted(rows[y])
        runs = []
        for x in xs:
            if runs and x == runs[-1][1] + 1:
                runs[-1][1] = x
            else:
                runs.append([x, x])
        single = len(runs) == 1
        for x0, x1 in runs:
            prev = ranges[-1] if ranges else None
            if single and prev and prev[3] and prev[1] == (x0, x1) and prev[2] == y - 1:
                prev[2] = y
            else:
                ranges.append([y, (x0, x1), y, single])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for y0, (x0, x1), y1, _ in ranges]
    )


def create_program_descriptor(device, input_tensor, weight_tensor, bias_tensor, output_tensor, plan, grid):
    p = plan
    bm, bk, bn = p["block_m_tiles"], p["block_k_tiles"], p["block_n_tiles"]
    has_bias = p["has_bias"]
    kx = p.get("k_exchange", False)  # sharded R4/R5 placement (shard input, shard output)
    act_mcast = p.get("act_mcast", False)  # K exchange across > 1 channel slice
    weight_mcast = p.get("weight_mcast", False)  # BLOCK: one DRAM weight read per grid column
    dw = p["depthwise"]  # R11: tap-weighted eltwise accumulation instead of the matmul
    weight_block_tiles = p["weight_k_rows"] * bn
    # Tilized in0 resident in cb_act_tiles: the m-block (dense) or DW_ACT_TILE_ROWS tile-rows (R11).
    act_tile_pages = (p["act_tile_rows"] or bm) * bk
    act_dtype = rm_dtype(input_tensor.dtype)  # the RM im2col pipeline's format
    w_dtype = weight_tensor.dtype
    tile_in = p.get("tile_in", False)  # TILE activation in and out (techniques 1 / 2)
    T_in = tile_bytes(input_tensor.dtype)
    T_out = tile_bytes(output_tensor.dtype) if tile_in else p["T"]  # output page: tile, or RM tile-row slab
    c_tiles = p.get("win_c_tiles", 0)
    direct_in0 = p.get("direct_in0", False)  # R9: shard tiles are in0 (no window / im2col / tilize)
    # R9: the RM pipeline's CBs are idle (one page each); in0 tiles keep the input's format.
    rm_pages = (lambda n: 1) if direct_in0 else (lambda n: n)
    in0_dtype, T_in0 = (input_tensor.dtype, T_in) if direct_in0 else (act_dtype, p["T"])
    eb = p["elem_bytes"]
    # BLOCK COL_MAJOR: the transposed grid — a K-exchange group (one shard row) is a grid
    # column and a weight-multicast group (one channel slice) is a grid row.
    col_major = kx and _block_col_major(input_tensor.memory_config())
    act_order = ttnn.McastCoreOrder.ColumnMajor if col_major else ttnn.McastCoreOrder.RowMajor
    is_height = kx and p["memory_layout"] == _ML.HEIGHT_SHARDED
    # Weight multicast groups: BLOCK = one channel slice (a grid column / row for COL_MAJOR);
    # HEIGHT = all compute cores in shard order, sender = core 0 (row-major first).
    weight_order = ttnn.McastCoreOrder.RowMajor if (col_major or is_height) else ttnn.McastCoreOrder.ColumnMajor
    math_fidelity, fp32_acc, _, acc_dtype = compute_policy(p["dtype"], p["weight_dtype"])

    # ---- Core map: (core, n-slice index, m-row index) ----
    if kx:
        core_map = _sharded_core_map(p, input_tensor.memory_config())
    else:
        core_map = _interleaved_core_map(p, grid)
    cores = _core_range_set([c for c, _, _ in core_map])
    # Rotating multicast groups must be rectangles: a WIDTH core set spanning grid rows
    # multicasts over its bounding rectangle, the extra cores being receive-only sinks.
    sink_cores = []
    mcast_cores = cores
    # R3 resident weights: multicast to every input core (fewest rectangles; the output-aligned
    # compute set can be gappy); input cores holding no output shard only receive (sinks).
    weight_group = cores
    weight_sender = 0
    if is_height and p.get("weights_resident", False):
        # The bounding rectangle of the input cores: one multicast rectangle (a ragged last grid
        # row would otherwise make two, and every rectangle re-sends the whole weight slice).
        bb = _core_range_set(height_shard_cores(input_tensor.memory_config(), p["rows_in"])).bounding_box()
        rect = [ttnn.CoreCoord(x, y) for y in range(bb.start.y, bb.end.y + 1) for x in range(bb.start.x, bb.end.x + 1)]
        weight_group = _core_range_set(rect)
        used = {(c.x, c.y) for c, _, _ in core_map}
        sink_cores = [c for c in rect if (c.x, c.y) not in used]
        mcast_cores = weight_group
        first = core_map[0][0]
        weight_sender = next(i for i, c in enumerate(rect) if (c.x, c.y) == (first.x, first.y))
    # The multicast helper covers at most 3 receiver rectangles; beyond that each core reads
    # its own weights (same writer path, weight_mcast off).
    weight_mcast = weight_mcast and len(weight_group.ranges()) <= MCAST_MAX_RECTANGLES
    # BLOCK streamed weights: rotate the column's DRAM reader / multicast sender over its rows.
    weight_rotating = (
        WEIGHT_ROTATING_SENDER
        and weight_mcast
        and p["memory_layout"] == _ML.BLOCK_SHARDED
        and weight_block_tiles * p["T_w"] <= WEIGHT_ROTATE_MAX_RATIO * act_tile_pages * p["T"]
        if kx
        else False
    )
    if is_height and not weight_mcast:  # no multicast: every compute core reads its own weights
        sink_cores, mcast_cores = [], cores
    if act_mcast:
        bb = cores.bounding_box()
        rect = [ttnn.CoreCoord(x, y) for y in range(bb.start.y, bb.end.y + 1) for x in range(bb.start.x, bb.end.x + 1)]
        real = [c for c, _, _ in core_map]
        if p["memory_layout"] == _ML.WIDTH_SHARDED and len(rect) > len(real):
            if [(c.x, c.y) for c in rect[: len(real)]] != [(c.x, c.y) for c in real]:
                raise UnsupportedAxisValue("conv2d_nhwc: WIDTH shard cores must be a row-major prefix of their grid")
            sink_cores = rect[len(real) :]
            mcast_cores = _core_range_set(rect)
        elif len(rect) != len(real):
            raise UnsupportedAxisValue("conv2d_nhwc: BLOCK shard grid must be a rectangle")

    # ---- Circular buffers (sizes derive from the knobs) ----
    src_bytes = p["src_bytes"]  # one window / halo stick slot
    cbs = [
        # R0: the DRAM window; R4/R5: the halo sticks (window minus the resident shard).
        # TILE: the untilized window (compute -> reader), one tile-row = c_tiles RM tile pages.
        _cb(CB_ACT_WINDOW, act_dtype, p["T"], rm_pages(p["win_tile_rows"] * c_tiles), cores)
        if tile_in
        else _cb(CB_ACT_WINDOW, act_dtype, src_bytes, p["window_sticks_max"], cores),
        _cb(CB_ACT_RM, act_dtype, p["T"], rm_pages(p["act_rm_rows"] * bk), cores),
        # Weight multicast landing slot: same address on every weight-group core (incl. sinks).
        _cb(CB_WEIGHT_TILES, w_dtype, p["T_w"], p["weight_depth"] * weight_block_tiles, weight_group),
        # Multicast landing zone: same address on every group core (incl. sinks).
        # R9 HEIGHT: in0 is the shard itself (cb_in_shard): this CB is idle.
        _cb(
            CB_ACT_TILES,
            in0_dtype,
            T_in0,
            (p["act_recv_depth"] if act_mcast else 1) * act_tile_pages if (act_mcast or not direct_in0) else 1,
            mcast_cores if act_mcast else cores,
        ),
    ]
    if p["stage_stride"]:
        cbs.append(_cb(CB_STICK_STAGE, act_dtype, p["stage_stride"], 2 * p["stage_row_sticks"], cores))
    if act_mcast:
        cbs.append(_cb(CB_ACT_SEND, act_dtype, p["T"], rm_pages(p["act_send_depth"] * bm * bk), cores))
    if tile_in:
        # Window (R0) / halo (sharded) tile-rows in the input's format, reader -> compute untilize.
        cbs.append(_cb(CB_WIN_TILES, input_tensor.dtype, T_in, rm_pages(p["halo_tile_rows"] * c_tiles), cores))
    gather_roles = p.get("num_gather_roles", 1)
    if gather_roles > 1:
        cbs.append(_cb(CB_ACT_WINDOW2, act_dtype, src_bytes, p["window_sticks_max"], cores))
        cbs.append(_cb(CB_ACT_RM2, act_dtype, p["T"], p["act_rm_rows"] * bk, cores))
    if kx:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_IN_SHARD, input_tensor, core_ranges=cores))
        full_rows = p["core_m_tiles"] - (1 if p["out_tail"] else 0)
        if full_rows > 0:
            out_cb = ttnn.cb_descriptor_from_sharded_tensor(
                CB_OUT_SHARD, output_tensor, total_size=full_rows * bn * T_out, core_ranges=cores
            )
            # Tile-sized pages: one untilized tile-row (bn pages) = 32 output-shard sticks; TILE:
            # the output tiles themselves (TileRowMajor partials = the shard's tile order).
            out_cb.format_descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT_SHARD, data_format=output_tensor.dtype, page_size=T_out)
            ]
            cbs.append(out_cb)
        if p["out_tail"]:
            cbs.append(_cb(CB_OUT_RM, output_tensor.dtype, p["T"], bn, cores))
    elif tile_in:
        # R0 TILE output: bias / copy pack one row group (sb_h x bn tiles) at a time; the writer
        # drains per tile-row. Two row groups = double-buffered.
        cbs.append(_cb(CB_OUT_RM, output_tensor.dtype, T_out, max(OUT_RM_DEPTH, 2 * p["out_subblock_h"]) * bn, cores))
    else:
        cbs.append(_cb(CB_OUT_RM, output_tensor.dtype, p["T"], OUT_RM_DEPTH * bn, cores))
    partial_formats = [ttnn.CBFormatDescriptor(buffer_index=CB_PARTIALS, data_format=acc_dtype, page_size=p["T_acc"])]
    if has_bias:
        cbs.append(_cb(CB_BIAS_TILES, bias_tensor.dtype, p["T_w"], p["core_n_tiles"], cores))
        # cb_biased aliases cb_partials: one allocation, two FIFOs (in-place bias add).
        partial_formats.append(
            ttnn.CBFormatDescriptor(buffer_index=CB_BIASED, data_format=acc_dtype, page_size=p["T_acc"])
        )
    cbs.append(
        ttnn.CBDescriptor(total_size=bm * bn * p["T_acc"], core_ranges=cores, format_descriptors=partial_formats)
    )

    rows_in = p.get("rows_in", 1) if kx else 0
    reader_k_blocks = p["kblocks_per_slice"] if kx else p["num_k_blocks"]

    # ---- Reader (NCRISC): window / halo fetch + im2col gather ----
    reader_ct = [
        CB_ACT_WINDOW,
        CB_ACT_RM,
        bm,
        bk,
        reader_k_blocks,
        p["num_n_blocks"],
        p["band_tiles"],
        p["tap_width"],
        eb,
        p["window_stick_bytes"],  # bytes read per DRAM input stick (R0; depthwise: own channels)
        p["kH"],
        p["kW"],
        p["H"],
        p["W"],
        p["H_out"],
        p["W_out"],
        p["M"],
        1 if kx else 0,
        CB_IN_SHARD,
        p.get("shard_h", 1) if kx else 1,
        max(rows_in, 1),
        1 if dw else 0,
        bn,
        1 if p.get("tap_major", False) else 0,
        gather_roles,
        1 if tile_in else 0,
        CB_WIN_TILES,
        c_tiles,
        p["win_tile_rows"],
        1 if direct_in0 else 0,
        p["dilation"],
        p["stage_stride"],
        CB_STICK_STAGE,
        p["stage_row_sticks"],
        p["window_zero_bytes"],
        src_bytes,
        _div_up(p["C_in"], TILE),  # input tile columns (TILE DRAM window page index)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # The writer carries the same gather args (its own window / rm CBs) for the split gather.
    writer_gather_ct = [CB_ACT_WINDOW2, CB_ACT_RM2] + reader_ct[2:]

    # ---- Writer (BRISC): weights, bias, K-exchange multicast (R4/R5), output ----
    writer_ct = [
        CB_WEIGHT_TILES,
        CB_BIAS_TILES,
        CB_OUT_RM,
        bm,
        bk,
        bn,
        p["num_k_blocks"],
        p["num_n_blocks"],
        p["N_t"],
        p["core_n_tiles"],
        1 if has_bias else 0,
        p["M"],
        eb,
        1 if kx else 0,
        CB_ACT_SEND,
        CB_ACT_TILES,
        p.get("kblocks_per_slice", 1),
        1 if p.get("out_tail", False) else 0,
        1 if act_mcast else 0,
        1 if weight_mcast else 0,
        p["weight_depth"],
        p["weight_k_rows"],
        1 if p.get("weights_resident", False) else 0,
        p.get("weight_chunk_rows", 1),
        1 if tile_in else 0,
        p["C_out"],
        p.get("act_channels", 1),
        p["g_m"] if weight_rotating else 0,
        weight_slice_tiles(input_tensor.memory_config() if kx else None, p["C_in"], p["C_out"], p.get("groups", 1)),
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
        if has_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(writer_gather_ct)

    # ---- Compute ----
    compute_ct = [
        CB_ACT_RM,
        CB_WEIGHT_TILES,
        CB_BIAS_TILES,
        CB_ACT_TILES,
        CB_PARTIALS,
        CB_BIASED,
        CB_OUT_RM,
        bm,
        bk,
        bn,
        p["num_k_blocks"],
        p["num_n_blocks"],
        p["out_subblock_h"],
        p["out_subblock_w"],
        1 if has_bias else 0,
        p["core_n_tiles"],
        1 if kx else 0,
        CB_ACT_SEND,
        p.get("kblocks_per_slice", 1),
        CB_OUT_SHARD,
        1 if p.get("out_tail", False) else 0,
        1 if act_mcast else 0,
        p.get("act_send_depth", 1),
        1 if p["activation"] == "relu" else 0,
        1 if dw else 0,
        p["taps"],
        p.get("tilize_w", bk),
        1 if p.get("weights_resident", False) else 0,
        CB_ACT_RM2,
        gather_roles,
        1 if tile_in else 0,
        CB_WIN_TILES,
        CB_ACT_WINDOW,
        CB_IN_SHARD,
        c_tiles,
        p["win_tile_rows"],
        1 if direct_in0 else 0,
        p.get("act_tilize_lead", 0),
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    w_addr = weight_tensor.buffer_address()
    b_addr = bias_tensor.buffer_address() if has_bias else 0
    out_addr = output_tensor.buffer_address()
    NHW = p["N"] * p["H"] * p["W"]
    # R3: every input shard is a potential halo source (shard order over the grid).
    halo_cores = height_shard_cores(input_tensor.memory_config(), rows_in) if is_height else []
    dna_mode = int(_os.environ.get("DNA_MODE", "0"))
    dna_on = bool(
        dna_mode
        and act_mcast
        and p["memory_layout"] == _ML.BLOCK_SHARDED
        and not sink_cores
        and p.get("act_channels", 1) == 1
    )
    dna_rt_off = None
    dna_rd_off = None
    for xy, ni, mi in core_map:
        m_tile_start = mi * p["core_m_tiles"]
        m_tiles = min(p["core_m_tiles"], p["M_t"] - m_tile_start)
        num_m_blocks = _div_up(m_tiles, bm)
        n_tile_start = ni * p["core_n_tiles"]
        # R0 depthwise: the window holds only this core's channel tiles (byte offset into the stick).
        win_chan_offset = n_tile_start * TILE * eb if (dw and not kx) else 0
        if p.get("native_group"):
            # Native groups: the window holds the core's group's C_in/g channels of each stick.
            win_chan_offset = (n_tile_start // p["group_n_tiles"]) * p["tap_width"] * eb
        reader_args = [
            in_addr,
            m_tile_start,
            m_tiles,
            num_m_blocks,
            p["padding"],
            p["stride"],
            p["dilation"],
            win_chan_offset,
        ]
        if kx:
            # Resident stick range + the NoC coords of this column's input shards (halo sources).
            if p["memory_layout"] == _ML.WIDTH_SHARDED:
                own_lo, own_hi, col = 0, NHW, [xy]
            elif p["memory_layout"] == _ML.HEIGHT_SHARDED:
                c = p["out_shards"][mi]
                own_lo, own_hi = c * p["shard_h"], min(NHW, (c + 1) * p["shard_h"])
                col = halo_cores
            else:
                own_lo, own_hi = mi * p["shard_h"], min(NHW, (mi + 1) * p["shard_h"])
                start = input_tensor.memory_config().shard_spec.grid.bounding_box().start
                col = [_block_core(start, ni, r, col_major) for r in range(rows_in)]
            reader_args += [own_lo, own_hi]
            for c in col:
                v = device.worker_core_from_logical_core(c)
                reader_args += [v.x, v.y]
        tile_win = list(p["tile_windows"][mi]) if (tile_in and kx) else [0, 0, 0, 0]
        if tile_in and kx:
            reader_args += tile_win  # first tile-row, halo-before / own / halo-after tile-row counts
        reader_rt[xy.x][xy.y] = reader_args
        writer_rt[xy.x][xy.y] = [
            w_addr,
            b_addr,
            out_addr,
            m_tile_start,
            m_tiles,
            num_m_blocks,
            n_tile_start,
            ni,
            0,  # not a sink
            # weight_mcast: the row (rotating sender: block w is read by row w % g_m), or 1 on grid
            # row 0 (fixed sender: it reads the column's weights).
            mi if weight_rotating else (1 if mi == 0 else 0),
        ] + (reader_args if gather_roles > 1 else [])
        compute_rt[xy.x][xy.y] = [num_m_blocks, ni] + tile_win[1:]
        if dna_on:
            grp = sorted([(n2, c2) for c2, n2, m2 in core_map if m2 == mi], key=lambda t: t[0])
            vv = [device.worker_core_from_logical_core(c2) for _, c2 in grp]
            wl = list(writer_rt[xy.x][xy.y])
            if dna_rt_off is None:
                dna_rt_off = len(wl)
            assert dna_rt_off == len(wl)
            wl += [min(v.x for v in vv), min(v.y for v in vv), max(v.x for v in vv), max(v.y for v in vv)]
            for v in vv:
                wl += [v.x, v.y]
            writer_rt[xy.x][xy.y] = wl
            rl = list(reader_rt[xy.x][xy.y])
            if dna_rd_off is None:
                dna_rd_off = len(rl)
            assert dna_rd_off == len(rl)
            rl += wl[dna_rt_off : dna_rt_off + 4]
            reader_rt[xy.x][xy.y] = rl
    for xy in sink_cores:
        # Receive-only: every round of every m-block (WIDTH: one m range for the whole group).
        writer_rt[xy.x][xy.y] = [0, 0, 0, 0, p["M_t"], _div_up(p["M_t"], bm), 0, 0, 1, 0]

    if dna_on and dna_mode in (1, 2):
        reader_cfg = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
        writer_cfg = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
    else:
        reader_cfg, writer_cfg = ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=reader_cfg,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_writer.cpp"),
        core_ranges=mcast_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=writer_cfg,
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    if act_mcast:
        # K exchange: one channel, groups = one grid row (BLOCK) / all WIDTH cores, rotating
        # sender = the slice owner (phase = slice index, row-major order).
        mcast = ttnn.Mcast(
            device,
            # The writer (WriterConfigDescriptor) issues on NOC_1 (preferred DRAM-write NoC).
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
            mcast_cores,
            mcast_cores.num_cores() if sink_cores else p["g_n"],
            ttnn.McastRotatingSenderConfig(),
            act_order,
        )
        mcast.attach(descriptor, "act_mcast", [writer], 0)
        next_sem = mcast.next_semaphore_id()
    else:
        ttnn.attach_absent_mcast(writer, "act_mcast")
        next_sem = 0
    if act_mcast and p["act_channels"] > 1:
        # The odd rounds' channel: same groups and rotation, its own semaphores.
        mcast2 = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
            mcast_cores,
            p["g_n"],
            ttnn.McastRotatingSenderConfig(),
            act_order,
        )
        mcast2.attach(descriptor, "act_mcast2", [writer], next_sem)
        next_sem = mcast2.next_semaphore_id()
    else:
        ttnn.attach_absent_mcast(writer, "act_mcast2")
    if weight_mcast:
        # Weights: groups = grid columns (BLOCK, column-major) / all cores (HEIGHT, row-major),
        # fixed sender = the group's first core (writer RT `mi == 0`).
        w_mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
            weight_group,
            weight_group.num_cores() if weight_group is not cores else p["g_m"],
            ttnn.McastRotatingSenderConfig()
            if weight_rotating
            else ttnn.McastFixedSenderConfig(sender_index=weight_sender),
            weight_order,
        )
        w_mcast.attach(descriptor, "w_mcast", [writer], next_sem)
        next_sem = w_mcast.next_semaphore_id()
    else:
        ttnn.attach_absent_mcast(writer, "w_mcast")
    sems = list(descriptor.semaphores)
    if dna_on:
        # ack, flag A, flag B, the VALID source word, then the reader mailbox (post, src, dst, done, bytes)
        for k, init in enumerate([0, 0, 0, 1, 0, 0, 0, 0, 0]):
            sems.append(ttnn.SemaphoreDescriptor(id=next_sem + k, core_ranges=mcast_cores, initial_value=init))
        descriptor.semaphores = sems
        if dna_mode == 3:
            descriptor.cbs = list(descriptor.cbs) + [
                _cb(15, ttnn.uint32, 32, 1, mcast_cores),
                _cb(16, ttnn.uint32, 32, 1, mcast_cores),
            ]
    named = list(writer.named_compile_time_args)
    named += [
        ("dna_mode", dna_mode if dna_on else 0),
        ("dna_b16", int(_os.environ.get("DNA_B16", "8"))),
        ("dna_rt_offset", dna_rt_off or 0),
        ("dna_sem_ack", next_sem),
        ("dna_sem_fa", next_sem + 1),
        ("dna_sem_fb", next_sem + 2),
        ("dna_sem_one", next_sem + 3),
        ("dna_chunk", int(_os.environ.get("DNA_CHUNK", "8192"))),
        ("dna_sem_post", next_sem + 4),
        ("dna_alink", int(_os.environ.get("DNA_ALINK", "1"))),
        ("dna_cb_post", 15),
        ("dna_cb_done", 16),
    ]
    writer.named_compile_time_args = named
    reader.named_compile_time_args = list(reader.named_compile_time_args) + [
        ("dna_mode", dna_mode if dna_on else 0),
        ("dna_rd_rt", dna_rd_off or 0),
        ("dna_sem_post", next_sem + 4),
        ("dna_num_slices", p["g_n"] if dna_on else 1),
        ("dna_b16", int(_os.environ.get("DNA_B16", "8"))),
        ("dna_sem_fb", next_sem + 2),
        ("dna_sem_one", next_sem + 3),
        ("dna_cb_post", 15),
        ("dna_cb_done", 16),
        ("dna_fine", int(_os.environ.get("DNA_FINE", "1"))),
    ]
    descriptor.kernels = [reader, writer, compute]
    return descriptor
