# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm_adaln — per-row LayerNorm over the last dim with optional per-call adaLN
modulation ``out = norm(x) * weight + bias``.

One ``ttnn.generic_op`` dispatch per call. See ``op_design.md`` (Blocking Model is binding)
and ``l1_ledger.md`` in this directory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if len(shape) >= 2 and shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "affine": ["no_affine", "dynamic_wb"],
    "affine_dtype": [ttnn.bfloat16, "none"],
    "affine_layout": [ttnn.TILE_LAYOUT, "none"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Compute config
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """Fresh default config per call (never a shared module-level object)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Blocking model (op_design.md "Regime/knob selection") — pure host function
# ---------------------------------------------------------------------------

# Host constants: single source of truth for every knob the kernels and CBs derive from.
MIN_SLICE_TILES = 8
GATHER_TILE_CAP = 48
# select_blocking cost model (units: one streamed tile on the busiest core). C_PEER is the cost of one (row, peer) pair
# of the stats all-gather + combine: the one-shot exchange of a compact payload prefix (512 B fp32 / 256 B bf16 per row)
# plus one streamed-fold slot. Fitted on BH 12x10 by a forced-S sweep of the current kernels in both DEST modes
# (perf_experiments/worksplit_refit, fit.py): the measured best S brackets it to 0.53 < C_PEER < 0.89 (512x3072 keeps
# S=6 over S=12 above 0.44, 8192x3072 noaff keeps S=3 over S=4 above 0.53, 2304x3072 wb takes S=6 over S=3 below 0.89).
# Re-fit if the exchange or combine kernels change.
C_PEER = 0.7
# Stagger-aware term (perf_experiments/worksplit_restagger, 9th tournament Perf 3). On a one-row blocking that the read
# stagger covers (stagger_phases), phase 0's stats chain + output writes run under phase 1's reads, so with two phases
# about half of a row's stream leaves the critical path: K_STAGGER = 0.5 (a forced-S sweep under the staggered kernels
# brackets it 0.4..0.8 with C_PEER = 0.7). It applies only to rows of <= STAGGER_HIDE_TILES tiles (32 KB): a gated
# 64 KB+ row read is bandwidth-bound, and taking the one-row split there measured +4.5..8.3% (2304x1024 noaff S1,
# 2048x2048 noaff S1, 640x6144 wb S6) against HEAD's multi-row pick.
K_STAGGER = 0.5
STAGGER_HIDE_TILES = 16
BF16_TILE_BYTES = 2048
DEFAULT_L1_BUDGET_BYTES = 1 << 20  # pure-function default; the entry point passes l1_cb_budget_bytes(device)
INPUT_BLOCK_DEPTH = 1
OUTPUT_ROW_DEPTH = 2


# Stats all-gather schedule (writer exchange_block): one-shot unicast while each core's S-1 outgoing payload copies stay
# under this many bytes (the rotating handshaked multicast that used to cover larger payloads was deleted once the
# compact payload prefix put every feasible blocking under the bound). Measured on BH 12x10 (perf_experiments/exchange_oneshot):
# one-shot wins / is flat up to 44 KiB per core and regresses 3-11% at >= 56 KiB (S-1 unicast copies vs one multicast
# saturate the NoC row).
ONESHOT_MAX_BYTES = 48 * 1024


def exchange_oneshot(num_w_slices, block_height, fp32_dest_acc_en, acc_tile_bytes):
    """True when the stats all-gather uses the one-shot unicast schedule (needs S > 1)."""
    # compact_stats_payload: only the first acc_tile_bytes / 8 of each payload tile travels.
    payload = stats_payload_tiles(fp32_dest_acc_en) * block_height * (acc_tile_bytes // 8)
    return num_w_slices > 1 and (num_w_slices - 1) * payload <= ONESHOT_MAX_BYTES


def gather_depth(num_w_slices, block_height, num_blocks_per_core, fp32_dest_acc_en, acc_tile_bytes):
    """Blocks held by cb_gathered_stats: the one-shot schedule lets a peer run one block ahead, so multi-block
    one-shot double-buffers the landing CB."""
    multi = num_blocks_per_core > 1
    return 2 if multi and exchange_oneshot(num_w_slices, block_height, fp32_dest_acc_en, acc_tile_bytes) else 1


def stats_payload_tiles(fp32_dest_acc_en):
    """Tiles per row tile of the published / all-gathered stats payload (sizes cb_local_stats / cb_gathered_stats and
    the exchange). Only column 0 of a stat tile is meaningful, so the payload packs every per-row stat into the same
    SFPU lanes of columns 0/1 and faces 0/2 | 1/3 of ONE tile: fp32 DEST [r_hi, r_lo | M2_hi, M2_lo] (tf32-exact split,
    copied through srcA); 16-bit DEST [p, M2 | d] (the slice mean carried pivot-relative, d = mean(x - p): a
    bf16-accumulated r is off by ~|mean|*2^-8*sqrt(n)). The publish then compacts those stats into face 0 rows 0..7
    (compact_stats_payload), so the one-shot exchange moves only acc_tile_bytes / 8 of each payload tile."""
    return 1


def slice_widths(tensor_w_tiles, num_w_slices):
    """Per-slice widths in tiles (wide_groups ragged slices): the first Wt % S slices are one tile wider."""
    q, r = divmod(tensor_w_tiles, num_w_slices)
    return [q + (1 if s < r else 0) for s in range(num_w_slices)]


@dataclass(frozen=True)
class Blocking:
    num_w_slices: int
    num_row_groups: int
    core_h_tiles: int
    block_height: int
    block_width: int
    num_blocks_per_core: int


def l1_cb_budget_bytes(device):
    """Per-core L1 bytes available to this program's CBs: the CB region starts at the allocator's unreserved base
    (above the kernel-config ring), not at the KERNEL_CONFIG base that get_max_worker_l1_unreserved_size() measures
    from (that over-counts by the ring, ~70 KB on BH, and let select_blocking pick blocks that overflow L1)."""
    return ttnn._ttnn.reports.get_device_info(device).cb_limit


def wbp_tiles(block_width, fp32_dest_acc_en):
    """cb_wbp P tiles: 3 columns per tile; 16-bit DEST pads to an even count (its normalize window reads two)."""
    return -(-block_width // 3) if fp32_dest_acc_en else 2 * -(-block_width // 6)


def footprint_bytes(
    block_height,
    block_width,
    num_w_slices,
    *,
    has_affine=True,
    acc_tile_bytes=4096,
    fp32_dest_acc_en=True,
    num_blocks_per_core=1,
):
    """Per-core L1 footprint of every CB the program descriptor allocates (exact: the sum of their total_size;
    l1_ledger.md 'Total per-core footprint')."""
    bh, bw, S = block_height, block_width, num_w_slices
    t16 = BF16_TILE_BYTES * (
        INPUT_BLOCK_DEPTH * bh * bw  # cb_input_tiles
        + 1  # cb_reduce_scaler
        + bh  # cb_pivot
        + OUTPUT_ROW_DEPTH * bw  # cb_output_tiles
        + (
            2 * bw if has_affine else 0
        )  # cb_weight_rows / cb_bias_rows: resident w / b (the column multicast lands them)
        + (wbp_tiles(bw, fp32_dest_acc_en) if has_affine else 0)  # cb_wbp: w / b P tiles (SFPU affine, per call)
        + (0 if fp32_dest_acc_en else (2 if has_affine else 1))  # cb_nstats: 16-bit per-row rstd / c'' strip tiles
    )
    # Per-row-tile acc CBs (TA = acc_tile_bytes): cb_local_stats (spr = 1 payload tile per row), cb_gathered_stats
    # (spr * S * depth), cb_row_stats [m, rstd] (2). The row's [sum (x - p), sum (x - p)^2] accumulators stay in DEST
    # through the row reduce and publish (stats_publish_fuse: no cb_dev_sq_acc / cb_dev_sq).
    spr = stats_payload_tiles(fp32_dest_acc_en)
    depth = gather_depth(S, bh, num_blocks_per_core, fp32_dest_acc_en, acc_tile_bytes)
    ta = acc_tile_bytes * bh * (spr + spr * S * depth + 2)
    return t16 + ta


# read_stagger (8th tournament Perf 2, read_stagger_v2; 3 phases: 9th tournament Perf 3, stagger_3phase): when every
# core owns one tile row (core_h_tiles == 1) the op is a chip-wide serial chain (all cores read into the DRAM wall, run
# the stats chain, then all write). The row groups are split into phases; a phase-p core's reader starts only after its
# same-slice phase-(p-1) predecessor has landed its row, so earlier phases' stats chains + output writes overlap later
# phases' reads (perf_experiments/read_stagger_v2, stagger_3phase). Mechanism-derived conditions keep it non-negative:
# * The gated phases must be DRAM-bandwidth bound: a phase's bytes at the chip read rate must cover one uncontended row
#   read (fixed latency + row bytes at the single-core read rate), else serializing the phases lengthens the read wall
#   by a latency-bound row read (64/96x3072, 256x1024, 256x2048 regressed +2..21%). Constants measured on BH 12x10
#   (gated rows land ~0.94 us after release for 16 KB, ~1.3 us for 32 KB; a gated phase reads at ~400 GB/s).
# * Three phases (top-first thirds) need every phase one group above that minimum: each phase boundary idles the DRAM
#   for about one group's release / landing skew (192x3072, three 2-group phases: +3.7..5%; phases of exactly m groups
#   with the store-bound rule below: 2304x256 / 576x1024 / 384x1536 +4.2..5.2%, stagger_s2_margin). Where they apply they
#   shrink the last phase's store tail (288 / 320 / 512 / 640 x3072, 1024x1024: -2.4..-6%). No endpoint-row group may
#   sit in the last phase: its exchange would run inside every earlier phase's store burst.
# * A third phase also needs the two-phase plan's last phase to be store-bound: its cores emit output faster than the
#   DRAM takes it (cores x tile bytes / normalize ns per tile > STAGGER_READ_BW). There the smaller last phase shortens
#   its store burst, and the previous phase's burst no longer covers its exchange. Where that phase's output is paced
#   by its normalize (affine: c_wbp_prep + the affine normalize, ~2.8 us of compute per 16-tile row), the split only
#   delays the last phase's release by one more phase read wall (992x1024 wb fp32: last phase issues +0.96 us, chain
#   from issue -0.24 us, 3 phases +4.8% vs 2; perf_experiments/stagger_s2_margin).
# * Two phases (the fallback): phase 0 holds every group whose grid row hosts a NoC1 DRAM write endpoint (the banks'
#   NoC1 optimal-worker rows). Every NoC1 store to a bank converges on that row, so a gated group there runs its
#   latency-critical S-member exchange inside phase 0's store burst and stalls 2-4 us (224x3072 grid row 5, 320x3072
#   grid row 8). Phase 0's exchange runs before any core stores, so endpoint rows are safe there.
STAGGER_READ_BW_BYTES_PER_NS = 400.0
STAGGER_ROW_LAT_NS = 580.0
STAGGER_CORE_READ_BYTES_PER_NS = 45.0
# c_normalize ns per output tile, (affine, fp32 DEST): p50 on BH 12x10 at 16-tile rows (stagger_s2_margin zones).
# Measured before normalize_window_fixed (same round) cut normalize ~10%; the merged round was A/B'd with these values.
NORMALIZE_TILE_NS = {(True, True): 172.0, (True, False): 158.0, (False, True): 113.0, (False, False): 84.0}


def stagger_phases(G, S, grid_x, ep_rows, row_bytes, has_affine, fp32_dest_acc_en):
    """Phase -> row groups of the read stagger, or None when it does not apply.
    Three phases when they fit: contiguous top-first thirds of the groups, every endpoint-row group in the last phase
    swapped with the bottom non-endpoint group of the nearest earlier phase. Used when no endpoint row is left in the
    last phase and every phase holds at least m + 1 groups, m = the bandwidth-bound minimum (ceil(min_bytes / group
    bytes)): a phase boundary idles the DRAM for about one group's skew, so a phase must stay bandwidth bound one group
    short; and only when the two-phase plan's last phase is store-bound (or there is no two-phase plan). Else two
    phases: every endpoint-row group, then the top non-endpoint groups up to half, in phase 0; the remaining (bottom)
    non-endpoint groups in phase 1, which must be bandwidth bound. Phase sizes never grow."""
    row_of = [(g * S) // grid_x for g in range(G)]
    min_bytes = STAGGER_READ_BW_BYTES_PER_NS * (STAGGER_ROW_LAT_NS + row_bytes / STAGGER_CORE_READ_BYTES_PER_NS)
    m = math.ceil(min_bytes / (S * row_bytes))
    if G >= 3:
        phases = [[g for g in range(G) if (g * 3) // G == p] for p in range(3)]
        for i, g in enumerate(list(phases[2])):
            if row_of[g] in ep_rows:
                for p in (1, 0):
                    cand = [h for h in phases[p] if row_of[h] not in ep_rows]
                    if cand:
                        phases[p][phases[p].index(cand[-1])] = g
                        phases[2][i] = cand[-1]
                        break
        phases3 = [sorted(ph) for ph in phases]
        if any(row_of[g] in ep_rows for g in phases3[2]) or min(len(ph) for ph in phases3) < m + 1:
            phases3 = None
    else:
        phases3 = None
    ep = [g for g in range(G) if row_of[g] in ep_rows]
    order = ep + [g for g in range(G) if row_of[g] not in ep_rows]
    n0 = max(len(ep), (G + 1) // 2)
    phase0, phase1 = order[:n0], order[n0:]
    phases2 = None if not phase1 or len(phase1) * S * row_bytes < min_bytes else [phase0, phase1]
    out_bytes_per_ns = len(phase1) * S * BF16_TILE_BYTES / NORMALIZE_TILE_NS[(bool(has_affine), bool(fp32_dest_acc_en))]
    if phases3 is not None and (phases2 is None or out_bytes_per_ns > STAGGER_READ_BW_BYTES_PER_NS):
        return phases3
    return phases2


def select_blocking(
    tensor_h_tiles,
    tensor_w_tiles,
    grid_x,
    grid_y,
    *,
    has_affine=True,
    acc_tile_bytes=4096,
    fp32_dest_acc_en=True,
    l1_budget_bytes=DEFAULT_L1_BUDGET_BYTES,
    stagger_ep_rows=frozenset(),
) -> Blocking:
    def fits(bh, bw, S, num_blocks=1):
        fp = footprint_bytes(
            bh,
            bw,
            S,
            has_affine=has_affine,
            acc_tile_bytes=acc_tile_bytes,
            fp32_dest_acc_en=fp32_dest_acc_en,
            num_blocks_per_core=num_blocks,
        )
        if fp > l1_budget_bytes:
            return False
        return S == 1 or stats_payload_tiles(fp32_dest_acc_en) * S * bh <= GATHER_TILE_CAP

    # wide_groups (10th tournament Perf 4): S need not divide Wt — ragged slices (slice_widths: the first Wt % S slices
    # one tile wider). Every slice keeps >= MIN_SLICE_TILES tiles; the cost model and footprint use the widest slice.
    candidates = [
        s for s in range(1, grid_x + 1) if grid_x % s == 0 and (s == 1 or tensor_w_tiles // s >= MIN_SLICE_TILES)
    ]
    forced = os.environ.get("LN_ADALN_FORCE_SLICES", "")  # experiment hook; unset = model choice
    if forced:
        if int(forced) not in candidates:
            raise ValueError(f"layer_norm_adaln: LN_ADALN_FORCE_SLICES={forced} not in feasible {candidates}")
        candidates = [int(forced)]
    best = None
    for S in candidates:
        bw = -(-tensor_w_tiles // S)
        groups_max = (grid_x // S) * grid_y
        groups = min(tensor_h_tiles, groups_max)
        core_h = math.ceil(tensor_h_tiles / groups)
        bh = core_h
        if not fits(bh, bw, S):
            continue
        # streamed tiles + per-call w/b fetch & broadcast prep + per-row S-way exchange and redundant combine, minus
        # the share of the row's stream the read stagger hides (phase 0's chain + writes under phase 1's reads)
        staggered = (
            core_h == 1
            and stagger_phases(groups, S, grid_x, stagger_ep_rows, bw * BF16_TILE_BYTES, has_affine, fp32_dest_acc_en)
            is not None
        )
        cost = (
            core_h * bw
            + (bw if has_affine else 0)
            + C_PEER * S * core_h
            - (K_STAGGER * bw if staggered and bw <= STAGGER_HIDE_TILES else 0)
        )
        if best is None or cost < best[0]:  # strict: tie -> smaller S (iterated ascending)
            best = (cost, Blocking(S, groups, core_h, bh, bw, 1))
    if best is not None:
        return best[1]

    # Fallback: multi-block rows_only, then smallest S whose 1-row block fits.
    groups = min(tensor_h_tiles, grid_x * grid_y)
    core_h = math.ceil(tensor_h_tiles / groups)
    bh = core_h
    while bh >= 1 and not fits(bh, tensor_w_tiles, 1):
        bh -= 1
    if bh >= 1 and 1 in candidates:
        return Blocking(1, groups, core_h, bh, tensor_w_tiles, math.ceil(core_h / bh))
    for S in candidates:
        bw = -(-tensor_w_tiles // S)
        groups = min(tensor_h_tiles, (grid_x // S) * grid_y)
        core_h = math.ceil(tensor_h_tiles / groups)
        if fits(1, bw, S, num_blocks=core_h):
            return Blocking(S, groups, core_h, 1, bw, core_h)
    raise ValueError(
        f"layer_norm_adaln: no feasible blocking for {tensor_h_tiles}x{tensor_w_tiles} tiles "
        "(regime wslice_reread is not built)"
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _affine_axes(weight, bias):
    if weight is not None and bias is not None:
        return "dynamic_wb", weight.dtype, weight.layout
    return "no_affine", "none", "none"


def validate(input_tensor, weight=None, bias=None, *, compute_kernel_config=None):
    if (weight is None) != (bias is None):
        raise ValueError("layer_norm_adaln: weight and bias must be supplied together")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    affine, affine_dtype, affine_layout = _affine_axes(weight, bias)
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    shape = list(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"layer_norm_adaln: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"layer_norm_adaln: unsupported combination (refinement candidate): {exc}")

    # Structural checks outside the axis rectangle.
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("layer_norm_adaln: input must be DRAM interleaved")
    if affine == "dynamic_wb":
        W = shape[-1]
        for name, t in (("weight", weight), ("bias", bias)):
            if list(t.shape) != [1, 1, 1, W]:
                raise ValueError(f"layer_norm_adaln: {name} shape {list(t.shape)} must be [1, 1, 1, {W}]")
            if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
                raise ValueError(f"layer_norm_adaln: {name} must be bfloat16 TILE")
            mc = t.memory_config()
            if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
                raise ValueError(f"layer_norm_adaln: {name} must be DRAM interleaved")
        if bias.dtype != weight.dtype or bias.layout != weight.layout:
            raise ValueError("layer_norm_adaln: weight and bias must share dtype/layout")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm_adaln(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor = None,
    bias: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    validate(input_tensor, weight, bias, compute_kernel_config=compute_kernel_config)
    from .layer_norm_adaln_program_descriptor import create_program_descriptor

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program_descriptor = create_program_descriptor(input_tensor, weight, bias, output_tensor, epsilon, cfg)
    io = [input_tensor] + ([weight, bias] if weight is not None else []) + [output_tensor]
    return ttnn.generic_op(io, program_descriptor)
