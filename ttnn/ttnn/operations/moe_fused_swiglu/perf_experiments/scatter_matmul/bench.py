# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""`scatter_matmul` — an isolated mini-op: LOCAL L1 MATMUL + COLUMN REDUCE, and nothing else.

This is the FLOOR of `moe_fused_swiglu`'s phase 1 with the weight and activation streams removed BY
CONSTRUCTION rather than by ablation. Every core's activations `x` and its two weight blocks
`wg` / `wu` are L1-resident sharded tensors consumed through ZERO-COPY tensor-backed CBs
(`ttnn.cb_descriptor_from_sharded_tensor`), so there is NO DRAM traffic anywhere in the timed
region — not one `TensorAccessor` read. Each core computes its own K-slice's `[M, N]` gate and up
partials with `matmul_block`, and then the grid COLUMN reduces those two partial blocks down to its
root. All `NCOLS` columns run concurrently so the NoC contention is the real thing.

It does NOT touch the real op.

WHAT VARIES — three orthogonal axes
-----------------------------------
1. REDUCE SHAPE (the transport topology), `shape`:

   `mm_only`  the matmul and no reduce at all — the floor every shape is measured against.

   `scatter`  ALL-TO-ALL SLICE SCATTER + GATHER (what the op ships). The `T = M*N` tile block is cut
              into `W` disjoint uniform slices; core r owns slice r. Every core unicasts its slice-r
              contribution of gate AND up straight into core r's landing CBs; each owner reduces
              only its own slice over all K contributors; then each owner unicasts its two finished
              slices straight into the column root's output shard at the slice's tile offset. THE
              GATHER IS THE ASSEMBLY — no root-side copy, no root-side add.

   `scatter_dual`  the same shape with the two operands split across the two NoCs: the WRITER
              (BRISC = NOC_1) ships gate, the READER (NCRISC = NOC_0) ships up, on both the scatter
              leg and the gather leg.

   `tree`     the op's PRE-scatter shape: a Hillis-Steele doubling tree per column. Every node folds
              its children's whole `T`-tile blocks onto its own and forwards to its parent; the root
              ends with the sum. `slots` landing slots with PER-SLOT arrival counters (the
              `reduce_transport_shape` finding), invited `slots` deep.

   `direct`   STAR: every non-root core ships its whole `T`-tile block to the root and the root does
              all K-1 adds. Same `slots` machinery as `tree`, fan-in K-1.

   `ring`     the classic RING REDUCE-SCATTER: K-1 sequential rounds around the column, each core
              passing an accumulating `a`-tile chunk to its left neighbour, then one gather round
              into the root. Landing L1 is O(T), like `scatter`, but the rounds serialise.

2. ACCUMULATE MECHANISM (how the adds are done), `mech` — a SEPARATE axis from the shape:

   `addchain`     what the op ships: `copy` the first contributor into an accumulator, then in-place
                  `add<blk_in(acc), blk_in(land_i), blk_out(acc)>` per further contributor, with the
                  LAST add landing in a FRESH CB so the writer can never observe a mid-chain state.
   `pack_l1_acc`  the PACKER folds each contributor onto a resident accumulator in L1; the
                  accumulator is never unpacked. The accumulator MUST be bf16 — at bfp8 the packer's
                  L1-accumulate is a linear add on a shared-exponent block-float tile, which is a
                  correctness bug (a prior bench measured PCC 0.412), so this mechanism pays one
                  extra `copy` to convert the bf16 accumulator back to the bfp8 the transport wants.
                  That conversion pass is charged to the mechanism, not hidden.
   `dest_acc`     all NC contributors summed in a sticky bf16 DEST window (<= 8 tiles under this op's
                  fixed precision contract) and packed to the output ONCE. Needs every contributor
                  RESIDENT AT ONCE, which only the `scatter` / `scatter_dual` shapes provide.
   `pack_l1_pair` two contributors folded per DEST window with one `BinaryFpu` add, then one
                  L1-accumulating pack — half the packs of `pack_l1_acc`. Also bf16, also charged
                  the conversion.

   MECHANISM AVAILABILITY IS ITSELF A PROPERTY OF THE SHAPE. `tree` / `direct` hold only `slots`
   contributors resident at a time, so they can express `addchain` and `pack_l1_acc` but NOT
   `dest_acc` / `pack_l1_pair`. `ring` folds exactly one contributor per round, so only `addchain`.

3. GEOMETRY: `m_eff` (token tile-rows per block), `N` (hidden tiles per column), `KGROUPS` (column
   depth), `KR` (per-core K tiles).

MEASURED (blackhole_p150, 1350 MHz, 11 concurrent columns, medians of 2 post-JIT runs, every cell
correctness-gated on min PCC over all 11 column roots). ALL numbers below were taken with a PRIVATE
`TT_METAL_PROFILER_DIR` (see the test module) — $TT_METAL_HOME/generated/profiler is shared across
concurrent runs on this box and the loser of the teardown race silently gets no data at all.

MATH UTILISATION IS THE HEADLINE. FPU roofline per core = (2 matmuls x m x n x kr) tile-MACs x 16
cycles / 1350 MHz, counted from the loop bounds `matmul_block` actually issues. `mm_only` — the same
matmul with the reduce deleted — is the control: `mm_efficiency = roofline / T(mm_only)` says how
good the bare matmul is, and everything above it is collective.

FOCUS (K=8, m_eff=8, N=6, KR=28, T=48; roofline 31 858 ns = 2688 tile-MACs):

    shape / mech                    ns   reduce cost   math_util   pcc        L1/core
    mm_only (matmul alone)       35 121           0      90.7%     —          631K
    scatter_dual/ pack_l1_pair   44 113       8 992      72.2%     0.999889   769K   <-- WINNER
    ring        / addchain       44 981       9 860      70.8%     0.999759   745K
    scatter_dual/ dest_acc       46 293      11 172      68.8%     0.999888   745K
    scatter_dual/ addchain       47 067      11 946      67.7%     0.999759   758K
    scatter     / pack_l1_pair   48 944      13 823      65.1%     0.999889   769K
    scatter     / dest_acc       50 981      15 860      62.5%     0.999888   745K
    scatter     / addchain       52 117      16 996      61.1%     0.999759   758K   <-- what ships
    tree        / addchain s=1   52 587      17 466      60.6%     0.999842   937K
    scatter     / pack_l1_acc    54 328      19 207      58.6%     0.999888   769K
    tree        / addchain s=2   54 822      19 701      58.1%     0.999842  1039K
    direct      / addchain s=2   69 007      33 886      46.2%     0.999759  1039K
    direct      / addchain s=1   71 249      36 128      44.7%     0.999759   937K

  * THE BARE MATMUL IS ALREADY AT 90.7 % OF ROOFLINE — every ns above it is pure collective cost.
    mm_efficiency rises with T (60 % at T=2, 91 % at T=48, 92 % at T=96): the shortfall is fixed
    per-program cost, not a matmul inefficiency.
  * WITH THE OPERANDS RESIDENT AND ONLY THE REDUCE LEFT, MATH UTILISATION LANDS AT 61-75 %
    depending ONLY on the reduce shape — versus ~38 % for the real op. The shape is worth 27
    percentage points on its own (direct 45 % -> scatter 61 % -> winner 72 %), so the reduce shape
    IS a real lever; but even the best shape leaves ~28 % of the wall on the collective.

FIDELITY PROBE (an instrument, not a variant: bfp8 x bfp4 => LoFi 1 FPU pass, HiFi2 2, HiFi4 4;
reported as the fraction of ADDED FPU work that is EXPOSED, which is a LOWER BOUND on FPU time):

    cell / shape                     LoFi     HiFi2     HiFi4    exp2   exp4   disagree
    m8 N6  mm_only                 35 117    68 953   132 673   106 %  102 %      4 %
    m8 N6  scatter/addchain        52 239    85 751   149 573   105 %  102 %      3 %
    m8 N6  scatter_dual/pack_pair  44 088    77 935   141 685   106 %  102 %      4 %

  The two slopes agree to 3-4 %, so the method holds here. Exposure is 102-106 % for EVERY shape and
  cell: added FPU work lands 1:1 on the wall whether or not a collective is running, i.e. NOTHING
  overlaps the FPU in this mini-op. (Cross-check on the constant: one measured pass is 33 836 ns
  (HiFi2 slope) / 32 519 ns (HiFi4 slope) = 17.0 / 16.3 cycles per tile-MAC, so the 16-cycle
  roofline is right to within 2-6 % and `mm_only` sits at ~96 % of the MEASURED pass cost.)

THE TWO WINS ARE ORTHOGONAL AND ADDITIVE — the full 2 (transport) x 4 (accumulate) factorial at
five cells separates them cleanly. At T=48: dual-NoC alone -5 000 ns, pack_l1_pair alone -3 074 ns,
both -7 928 ns against a -8 074 ns sum of the singles (2 % interaction). At T=96: -9 355 / -4 635 /
-14 116 vs -13 990 (1 %). Transport dominates at large T, accumulate at small T. Neither is
arithmetic:
  * `scatter_dual`: gate rides NOC_1 (writer/BRISC), up rides NOC_0 (reader/NCRISC), on both the
    scatter and the gather leg. Identical bytes, identical transaction count, identical adds — the
    two operands stop queueing on one network's injection port. NoC OCCUPANCY.
  * `pack_l1_pair`: the K-contributor slice fold becomes ceil(K/2) `eltwise_chain` calls (one
    BinaryFpu add + one L1-accumulating pack each) instead of K, halving the PACK count and the
    per-call init/reconfig. CB/DEST ROUND TRIPS AND CALL COUNT.

WINNER PREDICATE — `scatter_dual` + `pack_l1_pair` vs `scatter` + `addchain`, no regression in any
of the 29 measured cells:
    m_eff{1,2,4,8} x N{2,4,6,8,12} at K=8 : -14.3 % (T=96) .. -28.9 % (T=2); the win GROWS as T
    shrinks, because the matmul floor shrinks faster than the collective's fixed rendezvous.
    KGROUPS{4,8,10}                      : K=4 -7.8..-9.8 %, K=8 -15.2..-20.5 %, K=10 -18.8..-23.7 %;
    the win GROWS with K, because the all-to-all leg it splits is the term that scales with K.
  It also has the SMALLEST residual (ns - mm_only) at every cell measured, which is the graduation
  criterion: 9 038 ns at T=48 and 15 405 ns at T=96, vs the shipped shape's 16 999 / 29 639.

COLUMN DEPTH IS ITSELF A LEVER: at m_eff=8/N=6 the winner's math utilisation is 77 % at K=4, 72 %
at K=8, 65 % at K=10 (KR 28/28/23). Deeper columns buy parallel K at a direct cost in utilisation.

RING is genuinely fast — it ties the winner at the largest cells (84 999 vs 84 857 ns at T=96) and
beats the shipped scatter by 5.3-14.3 % — but it is NOT the recommendation: it requires K | T
exactly, which is false for 12 of the 20 surface cells and for EVERY K=10 cell the op runs. Its
speed is the instructive part: it moves the same total bytes as the all-to-all but only ever to an
ADJACENT core, so the transport pays for HOP-bytes, not bytes.

TREE vs SCATTER is a near-wash here (52 587 vs 52 117 at the focus), NOT the 2.80x the sibling
`reduce_scatter_swiglu` bench measured. That bench's scatter also DISTRIBUTED THE SwiGLU EPILOGUE;
this mini-op has no epilogue. So the shipped scatter's win was the distributed epilogue, and the
reduce transport shape on its own barely separates tree from scatter.

L1 IS A SHAPE CONSTRAINT: landing L1 is O(T) for scatter/ring but O(slots*T) + O(T) accumulator +
O(T) send for tree/direct, so at m_eff=8 / N=12 (T=96) tree and direct do not fit at all while
scatter (1279K) and ring (1253K) do.

DIRECT (star to the root) loses everywhere: 1.61x the winner at the focus, math utilisation 45 %.

PRECISION CONTRACT — FROZEN, identical for every variant, never a lever:
math_fidelity=LoFi, math_approx_mode=True, fp32_dest_acc_en=False, dst_full_sync_en=False,
bfp8_pack_precise=True; `x` and the partials are bfloat8_b, the weights bfloat4_b, and every
slice accumulator that is not bf16-by-mechanism is bfloat8_b.

RAW LLK: `pack_l1_acc` / `pack_l1_pair` drive `pack_reconfig_l1_acc(1)` directly instead of
`eltwise_chain`'s own `L1Accumulation` OutputSpec field. That field is a MANY:1 reduce primitive —
with it enabled the chain PINS the pack address (`out_idx = base`, not `base + i_flat`), collapsing
all block positions onto one tile, which is the wrong cardinality for a per-position accumulate.
The raw toggle with `TileOffset::Set{0}` keeps the position walk and only changes overwrite->add.
This is the same mechanism the op's own `down` matmul already relies on via `packer_l1_acc`.
"""

from dataclasses import dataclass

import ttnn

TILE = 32
BFP8_TILE_BYTES = 1088
BFP4_TILE_BYTES = 576
BF16_TILE_BYTES = 2048
DEST_LIMIT_TILES = 8  # DEST_AUTO_LIMIT at fp32_dest_acc_en=False / half sync
ELTWISE_BLK = 8  # the op's graduated Perf-1 DEST window; SAME in every variant
L1_CAP_BYTES = 1_572_864
L1_BUDGET_BYTES = 1_380_000  # leave headroom for firmware / kernel binaries / dispatch

# ---- CB indices. 0..8 fixed; the landing CBs start at CB_LAND_G and are laid out by the host. ----
CB_X = 0  # tensor-backed: this core's resident activation block (M*KR bfp8 tiles)
CB_WG = 1  # tensor-backed: gate weights   (KR*N bfp4_b tiles)
CB_WU = 2  # tensor-backed: up weights     (KR*N bfp4_b tiles)
CB_PG = 3  # my gate partial  (T bfp8 tiles) — the matmul's output
CB_PU = 4  # my up partial
CB_ACCG = 5  # reduce accumulator, gate (bfp8, or bf16 for the packer-L1-accumulate mechanisms)
CB_ACCU = 6
CB_SENDG = 7  # the FRESH CB the finished sum lands in (writer-visible); `ring`: the round payload
CB_SENDU = 8
CB_OUT = 9  # tensor-backed on the ROOT-ROW output shard (2T bfp8 tiles: gate sum then up sum)
CB_LAND_G = 10  # landing CBs: gate slots [CB_LAND_G, CB_LAND_G + n_slots)
# CB_LAND_U = CB_LAND_G + n_slots

SEM_INVITE = 0  # receiver -> contributors: "my landing region is reserved, ship"
SEM_DATA0 = 1  # contributor -> receiver, PER LANDING SLOT (SEM_DATA0 + slot)
SEM_DATA1 = 2
SEM_GATHER = 3  # worker -> column root: "my finished slice landed in your output shard"
SEM_RING = 4  # ring: right neighbour -> me, "round s landed"
NUM_SEMAPHORES = 5

SHAPES = ("mm_only", "scatter", "scatter_dual", "tree", "direct", "ring")
MECHS = ("addchain", "pack_l1_acc", "dest_acc", "pack_l1_pair")
_BF16_MECHS = ("pack_l1_acc", "pack_l1_pair")


# ---------------------------------------------------------------------------
# Host plumbing
# ---------------------------------------------------------------------------


def _grid_cores(k, ncols):
    """Every worker core, in the shard order a ROW_MAJOR height-sharded tensor uses over the same
    CoreRangeSet: shard index = row * ncols + col."""
    return [(col, row) for row in range(k) for col in range(ncols)]


def _core_range(cores):
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))])


def _virtual(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _cb(cb_index, core_ranges, num_pages, page_bytes=BFP8_TILE_BYTES, dtype=ttnn.bfloat8_b):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_bytes)],
    )


def _kernel(source, core_ranges, runtime_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=[],
        runtime_args=runtime_args,
        config=config,
    )


CHIP_FREQ_MHZ = 1350.0  # measured by the realtime profiler sync on this box (1.34998 GHz)
LOFI_CYCLES_PER_TILE_MAC = 16  # the LoFi matmul roofline constant


def roofline_ns(geo):
    """FPU-roofline time for ONE core's share of this cell, in ns.

    Counted from the loop bounds `matmul_block` actually runs, not from a shape guess: each call
    issues in0_num_subblocks * in1_num_subblocks * out_subblock_h * out_subblock_w * in0_block_k *
    num_k_blocks = m * n * kr tile-MACs, and this kernel makes TWO such calls (gate and up). There
    is no ragged/pad narrowing in this bench, so every issued tile-MAC is real work."""
    cycles = 2 * geo.m * geo.n * geo.kr * LOFI_CYCLES_PER_TILE_MAC
    return cycles / CHIP_FREQ_MHZ * 1000.0  # cycles / MHz * 1000 = ns


def tile_macs(geo):
    return 2 * geo.m * geo.n * geo.kr


def compute_config(fidelity=None):
    """PRECISION CONTRACT — byte-identical to moe_fused_swiglu.default_compute_kernel_config().
    A FIXED input to every variant; never touched for speed.

    `fidelity` is an INSTRUMENT, not a lever. The bake-off runs entirely at the default LoFi; the
    only reason this is a parameter is the fidelity probe, which re-runs a cell at HiFi2/HiFi4 to
    measure how much ADDED FPU work lands on the wall (i.e. what fraction of the FPU work is
    exposed). Those runs are diagnostics and are never reported as candidate variants."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity if fidelity is not None else ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        bfp8_pack_precise=True,
    )


def _largest_divisor_le(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _ew(n):
    """The op's blocked EltwiseShape. The DEST window must DIVIDE the tile count — ELTWISE_BLK
    is only the cap, and a window that does not divide would straddle the CB end."""
    return f"EltwiseShape::tiles({n}, {_largest_divisor_le(n, ELTWISE_BLK)})"


@dataclass(frozen=True)
class Geo:
    """`ncols` independent `k`-deep reduce columns laid out exactly as the op's HGROUPS x KGROUPS
    worker grid. Every collective stays INSIDE its column; `ncols > 1` adds only the thing a
    single-column bench cannot see — the other columns' concurrent NoC traffic."""

    k: int  # KGROUPS: column depth
    m: int  # m_eff: token tile-rows per block
    n: int  # N: hidden tiles per column
    kr: int  # per-core K tiles
    ncols: int  # HGROUPS: concurrent columns

    @property
    def t(self):
        return self.m * self.n

    @property
    def workers(self):
        """Uniform slice plan: the largest divisor of T that is <= K. Uniform slices keep every
        slice CB at its natural page count (the ragged split needs lcm-sized CBs and measured
        catastrophically worse in the sibling reduce_scatter bench)."""
        return _largest_divisor_le(self.t, self.k)

    @property
    def a(self):
        return self.t // self.workers

    @property
    def x_tiles(self):
        return self.m * self.kr

    @property
    def w_tiles(self):
        return self.kr * self.n

    @property
    def core_range(self):
        return _core_range(_grid_cores(self.k, self.ncols))

    @property
    def root_range(self):
        return _core_range([(c, 0) for c in range(self.ncols)])

    def shard_index(self, col, row):
        return row * self.ncols + col


def hillis_steele_tree(k, root=0):
    """The op's SHIPPED tree (`_reduce_tree`), one column of `k` rows. Root fan-in is ceil(log2(k))
    because the accumulator (relative index 0) stays the SAME physical node at every doubling."""
    info = {}
    for y in range(k):
        r = (y - root) % k
        children, s = [], 1
        while s < k:
            if r % (2 * s) == 0 and r + s < k:
                children.append((root + r + s) % k)
            s *= 2
        parent = None if r == 0 else (root + r - (r & (-r))) % k
        info[y] = {"parent": parent, "children": children}
    return info


def plan(shape, geo, slots=1):
    """Everything the kernels and the L1 accounting need, derived once on the host.

    Returns a dict with:
      n_slots     landing slots per operand (scatter/ring: one per contributor; tree/direct: `slots`)
      slot_tiles  tiles per landing slot
      assigned    per-row slice size (0 = idle core)
      offsets     per-row slice tile offset in the T-tile block
      contribs    per-row list of (contributor_row, slot) that land on that row
      nc          contributors folded per reducing core
    """
    k, t, a, w = geo.k, geo.t, geo.a, geo.workers
    if shape == "mm_only":
        return dict(n_slots=0, slot_tiles=0, assigned=[0] * k, offsets=[0] * k, nc=0, tree=None)
    if shape in ("scatter", "scatter_dual"):
        assigned = [a] * w + [0] * (k - w)
        offsets = [i * a for i in range(w)] + [0] * (k - w)
        return dict(n_slots=k, slot_tiles=a, assigned=assigned, offsets=offsets, nc=k, tree=None)
    if shape == "ring":
        # K chunks, one per core; core r ends holding chunk (r-1) mod K complete.
        assigned = [a] * k
        offsets = [((r - 1) % k) * a for r in range(k)]
        return dict(n_slots=k - 1, slot_tiles=a, assigned=assigned, offsets=offsets, nc=1, tree=None)
    if shape == "direct":
        assigned = [0] * k
        assigned[0] = t
        return dict(n_slots=slots, slot_tiles=t, assigned=assigned, offsets=[0] * k, nc=k - 1, tree=None)
    if shape == "tree":
        tree = hillis_steele_tree(k)
        assigned = [t if len(tree[r]["children"]) or r == 0 else 0 for r in range(k)]
        return dict(
            n_slots=slots,
            slot_tiles=t,
            assigned=assigned,
            offsets=[0] * k,
            nc=max(len(tree[r]["children"]) for r in range(k)),
            tree=tree,
        )
    raise ValueError(f"unknown shape {shape!r}")


def _acc_bytes(mech):
    return BF16_TILE_BYTES if mech in _BF16_MECHS else BFP8_TILE_BYTES


def l1_bytes(shape, geo, mech="addchain", slots=1):
    """Per-core L1: the resident tensors + every scratch CB. The output shard lives only on the
    root row, so it is counted separately (and is what actually bounds the root)."""
    p = plan(shape, geo, slots)
    tensors = geo.x_tiles * BFP8_TILE_BYTES + 2 * geo.w_tiles * BFP4_TILE_BYTES
    scratch = 2 * geo.t * BFP8_TILE_BYTES  # cb_pg + cb_pu
    if shape == "mm_only":
        return tensors + scratch, tensors + scratch + 2 * geo.t * BFP8_TILE_BYTES
    landing = 2 * p["n_slots"] * p["slot_tiles"] * BFP8_TILE_BYTES
    if shape == "ring":
        send = 2 * 2 * geo.a * BFP8_TILE_BYTES  # ringout, double-buffered
        acc = 0
    else:
        send = (
            2 * p["slot_tiles"] * BFP8_TILE_BYTES
            if shape not in ("scatter", "scatter_dual")
            else 2 * geo.a * BFP8_TILE_BYTES
        )
        acc = (
            0
            if mech == "dest_acc"
            else 2 * (geo.a if shape in ("scatter", "scatter_dual") else geo.t) * _acc_bytes(mech)
        )
    worker = tensors + scratch + landing + send + acc
    root = worker + 2 * geo.t * BFP8_TILE_BYTES  # the output shard
    return worker, root


def feasible(shape, geo, mech="addchain", slots=1):
    """(ok, reason). Everything the geometry must satisfy for this (shape, mech) to be expressible."""
    if geo.t < 1:
        return False, "empty block"
    if shape in ("scatter", "scatter_dual") and geo.workers < 2:
        return False, f"T={geo.t} gives only {geo.workers} uniform slices; a scatter needs >= 2"
    if shape == "ring":
        if geo.t % geo.k:
            return False, f"ring needs K={geo.k} equal chunks; T={geo.t} is not divisible"
        if geo.k < 3:
            return False, "ring degenerates below K=3"
    if shape in ("tree", "direct") and geo.k < 2:
        return False, "no cross-core reduce at K=1"
    if mech in ("dest_acc", "pack_l1_pair") and shape not in ("scatter", "scatter_dual"):
        return False, f"{mech} needs every contributor resident at once; {shape} holds {slots}"
    if shape == "ring" and mech != "addchain":
        return False, "ring folds exactly one contributor per round"
    if shape == "mm_only" and mech != "addchain":
        return False, "no reduce to vary"
    n_red = geo.a if shape in ("scatter", "scatter_dual") else geo.t
    if mech in ("dest_acc", "pack_l1_pair") and _largest_divisor_le(n_red, DEST_LIMIT_TILES) < 1:
        return False, "no legal DEST window"
    worker, root = l1_bytes(shape, geo, mech, slots)
    if root > L1_BUDGET_BYTES:
        return False, f"L1 {root} B > budget {L1_BUDGET_BYTES} B"
    return True, ""


# ===========================================================================
# Kernel sources — generated per (shape, mech, geometry) so every constant is a
# literal and there is no compile-time-arg plumbing to get wrong.
# ===========================================================================

_DF_INCLUDES = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
"""

# ---- READER (NCRISC / NOC_0) ------------------------------------------------
# Common prologue on EVERY shape: expose the three resident, tensor-backed operand CBs. This is the
# whole "no DRAM traffic" claim in three lines — the CBs ARE the sharded L1 buffers, so making the
# data visible to the unpacker costs one counter update each and moves ZERO bytes.
_READER_PROLOGUE = """
    cb_reserve_back({cb_x}, {x_tiles});  cb_push_back({cb_x}, {x_tiles});
    cb_reserve_back({cb_wg}, {w_tiles}); cb_push_back({cb_wg}, {w_tiles});
    cb_reserve_back({cb_wu}, {w_tiles}); cb_push_back({cb_wu}, {w_tiles});
"""


def _reader_source(shape, geo, p, mech, slots):
    """RT args: 0 my_row, 1 is_root, 2 assigned, 3 offset, 4 root_vx, 5 root_vy, 6 out_addr,
    7 n_peer, 8.. (vx, vy) pairs."""
    body = _READER_PROLOGUE.format(cb_x=CB_X, cb_wg=CB_WG, cb_wu=CB_WU, x_tiles=geo.x_tiles, w_tiles=geo.w_tiles)
    land_g, land_u = CB_LAND_G, CB_LAND_G + max(p["n_slots"], 1)
    if shape == "mm_only":
        pass
    elif shape in ("scatter", "scatter_dual"):
        # Every contributor increments SEM_DATA0 once per operand it ships. `scatter` ships both on
        # the writer (1 inc/contributor); `scatter_dual` splits them across the two NoCs, so each
        # contributor increments TWICE. The receiver's threshold is the only thing that changes.
        arrivals = 2 * geo.k if shape == "scatter_dual" else geo.k
        gather_arrivals = 2 * geo.workers if shape == "scatter_dual" else geo.workers
        dual_send = ""
        if shape == "scatter_dual":
            db = 8 + 2 * geo.k  # dest quadruples follow the K peer (vx, vy) pairs
            dual_send = f"""
    // --- NOC_0's half of the scatter: the `up` operand. The gate half rides NOC_1 on the writer,
    // so both networks inject concurrently instead of queueing the two operands on NOC_1. ---
    noc_semaphore_wait_min(invite_ptr, {geo.k});
    cb_wait_front({CB_PU}, {geo.t});
    {{
        const uint32_t usrc = get_read_ptr({CB_PU});
        const uint32_t udst = get_write_ptr({land_u} + my_row);
        const uint32_t sem_data = static_cast<uint32_t>(get_semaphore({SEM_DATA0}));
        for (uint32_t d = 0; d < {geo.k}; ++d) {{
            if (get_arg_val<uint32_t>({db} + 4 * d + 3) == 0) {{ continue; }}
            noc_async_write(usrc + get_arg_val<uint32_t>({db} + 4 * d + 2) * {BFP8_TILE_BYTES},
                            get_noc_addr(get_arg_val<uint32_t>({db} + 4 * d + 0),
                                         get_arg_val<uint32_t>({db} + 4 * d + 1), udst),
                            {geo.a * BFP8_TILE_BYTES});
        }}
        noc_async_write_barrier();
        for (uint32_t d = 0; d < {geo.k}; ++d) {{
            if (get_arg_val<uint32_t>({db} + 4 * d + 3) == 0) {{ continue; }}
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>({db} + 4 * d + 0),
                                           get_arg_val<uint32_t>({db} + 4 * d + 1), sem_data), 1);
        }}
        noc_async_atomic_barrier();
        cb_pop_front({CB_PU}, {geo.t});
    }}
"""
        dual_gather = ""
        if shape == "scatter_dual":
            dual_gather = f"""
    if (assigned) {{
        cb_wait_front({CB_SENDU}, {geo.a});
        noc_async_write(get_read_ptr({CB_SENDU}),
                        get_noc_addr(root_vx, root_vy, out_addr + ({geo.t} + offset) * {BFP8_TILE_BYTES}),
                        {geo.a * BFP8_TILE_BYTES});
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_vx, root_vy,
                                       static_cast<uint32_t>(get_semaphore({SEM_GATHER}))), 1);
        noc_async_atomic_barrier();
        cb_pop_front({CB_SENDU}, {geo.a});
    }}
"""
        body += f"""
    const uint32_t my_row = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t assigned = get_arg_val<uint32_t>(2);
    const uint32_t offset = get_arg_val<uint32_t>(3);
    const uint32_t root_vx = get_arg_val<uint32_t>(4);
    const uint32_t root_vy = get_arg_val<uint32_t>(5);
    const uint32_t out_addr = get_arg_val<uint32_t>(6);
    (void)offset; (void)root_vx; (void)root_vy; (void)out_addr;
    volatile tt_l1_ptr uint32_t* invite_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_INVITE})));
    (void)invite_ptr;

    // Reserve every landing slot BEFORE inviting: a contributor's landing address is its OWN
    // get_write_ptr of the same CB index (identical CB layout on every core, whole-CB push, so the
    // write pointer is always the CB base) and that is only a valid proxy once the region is free.
    if (assigned) {{
        for (uint32_t i = 0; i < {geo.k}; ++i) {{
            cb_reserve_back({land_g} + i, {geo.a});
            cb_reserve_back({land_u} + i, {geo.a});
        }}
    }}
    for (uint32_t p = 0; p < {geo.k}; ++p) {{
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(8 + 2 * p), get_arg_val<uint32_t>(8 + 2 * p + 1),
                         static_cast<uint32_t>(get_semaphore({SEM_INVITE}))), 1);
    }}
    noc_async_atomic_barrier();
{dual_send}
    if (assigned) {{
        volatile tt_l1_ptr uint32_t* data_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_DATA0})));
        noc_semaphore_wait_min(data_ptr, {arrivals});
        for (uint32_t i = 0; i < {geo.k}; ++i) {{
            cb_push_back({land_g} + i, {geo.a});
            cb_push_back({land_u} + i, {geo.a});
        }}
    }}
{dual_gather}
    if (is_root) {{
        volatile tt_l1_ptr uint32_t* gather_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_GATHER})));
        noc_semaphore_wait_min(gather_ptr, {gather_arrivals});
    }}
"""
    elif shape == "ring":
        body += f"""
    const uint32_t my_row = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    (void)my_row;
    // All K-1 round slots are reserved UP FRONT, so the ring needs no per-round slot-free invite:
    // one initial invite to my right neighbour (the core that ships to me) and then a monotone
    // arrival counter is the entire protocol.
    for (uint32_t i = 0; i < {geo.k - 1}; ++i) {{
        cb_reserve_back({land_g} + i, {geo.a});
        cb_reserve_back({land_u} + i, {geo.a});
    }}
    noc_semaphore_inc(
        get_noc_addr(get_arg_val<uint32_t>(8), get_arg_val<uint32_t>(9),
                     static_cast<uint32_t>(get_semaphore({SEM_INVITE}))), 1);
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* ring_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_RING})));
    for (uint32_t s = 0; s < {geo.k - 1}; ++s) {{
        noc_semaphore_wait_min(ring_ptr, s + 1);
        cb_push_back({land_g} + s, {geo.a});
        cb_push_back({land_u} + s, {geo.a});
    }}
    if (is_root) {{
        volatile tt_l1_ptr uint32_t* gather_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_GATHER})));
        noc_semaphore_wait_min(gather_ptr, {geo.k});
    }}
"""
    else:  # tree / direct — `slots` landing slots, PER-SLOT arrival counters, invited `slots` deep
        body += f"""
    const uint32_t n_peer = get_arg_val<uint32_t>(7);
    volatile tt_l1_ptr uint32_t* data_ptr[2] = {{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_DATA0}))),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_DATA1})))}};
    const uint32_t sem_invite = static_cast<uint32_t>(get_semaphore({SEM_INVITE}));
    uint32_t seen[2] = {{0, 0}};
    // Prime `slots` invites, then run one-in-one-out: the cb_reserve_back for contributor c+slots
    // is the back-pressure (it blocks until compute has popped slot `c % slots`), and the PER-SLOT
    // counter lets slot s be published the moment slot s lands instead of waiting for a whole wave.
    for (uint32_t c = 0; c < n_peer && c < {slots}; ++c) {{
        cb_reserve_back({land_g} + c, {geo.t});
        cb_reserve_back({land_u} + c, {geo.t});
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(8 + 2 * c), get_arg_val<uint32_t>(8 + 2 * c + 1),
                                       sem_invite), 1);
    }}
    for (uint32_t c = 0; c < n_peer; ++c) {{
        const uint32_t slot = c % {slots};
        seen[slot] += 1;
        noc_semaphore_wait_min(data_ptr[slot], seen[slot]);
        cb_push_back({land_g} + slot, {geo.t});
        cb_push_back({land_u} + slot, {geo.t});
        const uint32_t nxt = c + {slots};
        if (nxt < n_peer) {{
            cb_reserve_back({land_g} + slot, {geo.t});
            cb_reserve_back({land_u} + slot, {geo.t});
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(8 + 2 * nxt),
                                           get_arg_val<uint32_t>(8 + 2 * nxt + 1), sem_invite), 1);
            // The per-invite flush is NOT optional on this path, and it is not cosmetic: dropping it
            // in favour of one flush per kernel (the idiom the scatter/ring readers use, where every
            // invite is issued in one burst before any wait) HANGS this reader even at the focus
            // geometry. The difference is that here the invite is issued from INSIDE a loop that
            // immediately re-enters `noc_semaphore_wait_min` on a different semaphore, so the
            // increment has to be known-flushed before the RISC parks. It costs an atomic round trip
            // per contributor — a real, disclosed cost of the recycling-slot topology, and part of
            // why `tree` and `direct` measure the way they do.
            noc_async_atomic_barrier();
        }}
    }}
    noc_async_atomic_barrier();
"""
    return _DF_INCLUDES + "\nvoid kernel_main() {\n" + body + "\n}\n"


# ---- WRITER (BRISC / NOC_1) -------------------------------------------------


def _writer_source(shape, geo, p, mech, slots):
    """RT args: 0 my_row, 1 is_root, 2 assigned, 3 offset, 4 root_vx, 5 root_vy, 6 out_addr,
    7 parent_vx, 8 parent_vy, 9 my_slot, 10.. per-destination quadruples (vx, vy, dst_off, dst_a)."""
    land_g, land_u = CB_LAND_G, CB_LAND_G + max(p["n_slots"], 1)
    pb = BFP8_TILE_BYTES
    head = f"""
    const uint32_t my_row = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t assigned = get_arg_val<uint32_t>(2);
    const uint32_t offset = get_arg_val<uint32_t>(3);
    const uint32_t root_vx = get_arg_val<uint32_t>(4);
    const uint32_t root_vy = get_arg_val<uint32_t>(5);
    const uint32_t out_addr = get_arg_val<uint32_t>(6);
    (void)my_row; (void)is_root; (void)assigned; (void)offset;
    (void)root_vx; (void)root_vy; (void)out_addr;
"""
    if shape == "mm_only":
        # The matmul's outputs must still be drained so the CB accounting matches every other shape.
        body = f"""
    cb_wait_front({CB_PG}, {geo.t});
    cb_wait_front({CB_PU}, {geo.t});
    cb_pop_front({CB_PG}, {geo.t});
    cb_pop_front({CB_PU}, {geo.t});
"""
    elif shape in ("scatter", "scatter_dual"):
        if shape == "scatter_dual":
            # NOC_1 gathers only `gate`; the reader gathers `up` on NOC_0. Each worker therefore
            # signals the root TWICE, which is why the root's threshold is 2 * workers.
            gather = f"""
        cb_wait_front({CB_SENDG}, {geo.a});
        noc_async_write(get_read_ptr({CB_SENDG}), get_noc_addr(root_vx, root_vy, out_addr + offset * {pb}),
                        {geo.a * pb});
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_vx, root_vy, static_cast<uint32_t>(get_semaphore({SEM_GATHER}))), 1);
        noc_async_atomic_barrier();
        cb_pop_front({CB_SENDG}, {geo.a});
"""
        else:
            gather = f"""
        cb_wait_front({CB_SENDG}, {geo.a});
        cb_wait_front({CB_SENDU}, {geo.a});
        noc_async_write(get_read_ptr({CB_SENDG}), get_noc_addr(root_vx, root_vy, out_addr + offset * {pb}),
                        {geo.a * pb});
        noc_async_write(get_read_ptr({CB_SENDU}),
                        get_noc_addr(root_vx, root_vy, out_addr + ({geo.t} + offset) * {pb}), {geo.a * pb});
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(root_vx, root_vy, static_cast<uint32_t>(get_semaphore({SEM_GATHER}))), 1);
        noc_async_atomic_barrier();
        cb_pop_front({CB_SENDG}, {geo.a});
        cb_pop_front({CB_SENDU}, {geo.a});
"""
        if shape == "scatter_dual":
            send = f"""
    // NOC_1's half: `gate` only (the reader ships `up` on NOC_0).
    cb_wait_front({CB_PG}, {geo.t});
    const uint32_t gsrc = get_read_ptr({CB_PG});
    const uint32_t gdst = get_write_ptr({land_g} + my_row);
    for (uint32_t d = 0; d < {geo.k}; ++d) {{
        const uint32_t da = get_arg_val<uint32_t>(10 + 4 * d + 3);
        if (da == 0) {{ continue; }}
        const uint32_t vx = get_arg_val<uint32_t>(10 + 4 * d + 0);
        const uint32_t vy = get_arg_val<uint32_t>(10 + 4 * d + 1);
        const uint32_t doff = get_arg_val<uint32_t>(10 + 4 * d + 2);
        noc_async_write(gsrc + doff * {pb}, get_noc_addr(vx, vy, gdst), {geo.a * pb});
    }}
    noc_async_write_barrier();
"""
            pop = f"    cb_pop_front({CB_PG}, {geo.t});\n"
        else:
            send = f"""
    cb_wait_front({CB_PG}, {geo.t});
    cb_wait_front({CB_PU}, {geo.t});
    const uint32_t gsrc = get_read_ptr({CB_PG});
    const uint32_t usrc = get_read_ptr({CB_PU});
    const uint32_t gdst = get_write_ptr({land_g} + my_row);
    const uint32_t udst = get_write_ptr({land_u} + my_row);
    for (uint32_t d = 0; d < {geo.k}; ++d) {{
        const uint32_t da = get_arg_val<uint32_t>(10 + 4 * d + 3);
        if (da == 0) {{ continue; }}
        const uint32_t vx = get_arg_val<uint32_t>(10 + 4 * d + 0);
        const uint32_t vy = get_arg_val<uint32_t>(10 + 4 * d + 1);
        const uint32_t doff = get_arg_val<uint32_t>(10 + 4 * d + 2);
        // ONE coalesced transaction per operand per destination: a slice is a CONTIGUOUS tile range
        // in the `m * N + n` output layout the matmul emits (out_subblock_h == 1, TileRowMajor).
        noc_async_write(gsrc + doff * {pb}, get_noc_addr(vx, vy, gdst), {geo.a * pb});
        noc_async_write(usrc + doff * {pb}, get_noc_addr(vx, vy, udst), {geo.a * pb});
    }}
    noc_async_write_barrier();
"""
            pop = f"    cb_pop_front({CB_PG}, {geo.t});\n    cb_pop_front({CB_PU}, {geo.t});\n"
        body = f"""
    volatile tt_l1_ptr uint32_t* invite_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_INVITE})));
    noc_semaphore_wait_min(invite_ptr, {geo.k});
{send}
    {{
        const uint32_t sem_data = static_cast<uint32_t>(get_semaphore({SEM_DATA0}));
        for (uint32_t d = 0; d < {geo.k}; ++d) {{
            if (get_arg_val<uint32_t>(10 + 4 * d + 3) == 0) {{ continue; }}
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(10 + 4 * d + 0),
                                           get_arg_val<uint32_t>(10 + 4 * d + 1), sem_data), 1);
        }}
        noc_async_atomic_barrier();
    }}
{pop}
    if (assigned) {{
        // THE GATHER IS THE ASSEMBLY: the finished slice goes straight into the root's output shard
        // at its own tile offset, so there is no root-side copy and no root-side add.
{gather}
    }}
"""
    elif shape == "ring":
        body = f"""
    volatile tt_l1_ptr uint32_t* invite_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_INVITE})));
    noc_semaphore_wait_min(invite_ptr, 1);
    const uint32_t left_vx = get_arg_val<uint32_t>(7);
    const uint32_t left_vy = get_arg_val<uint32_t>(8);
    const uint32_t sem_ring = static_cast<uint32_t>(get_semaphore({SEM_RING}));
    // K-1 SEQUENTIAL rounds: round s ships the chunk this core has accumulated so far into the
    // left neighbour's slot s. This is the ring's whole cost model — the bytes are small but the
    // rounds cannot overlap, because round s+1's payload is round s's arrival plus a local add.
    for (uint32_t s = 0; s < {geo.k - 1}; ++s) {{
        cb_wait_front({CB_SENDG}, {geo.a});
        cb_wait_front({CB_SENDU}, {geo.a});
        noc_async_write(get_read_ptr({CB_SENDG}), get_noc_addr(left_vx, left_vy, get_write_ptr({land_g} + s)),
                        {geo.a * pb});
        noc_async_write(get_read_ptr({CB_SENDU}), get_noc_addr(left_vx, left_vy, get_write_ptr({land_u} + s)),
                        {geo.a * pb});
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(left_vx, left_vy, sem_ring), 1);
        noc_async_atomic_barrier();
        cb_pop_front({CB_SENDG}, {geo.a});
        cb_pop_front({CB_SENDU}, {geo.a});
    }}
    // The gather round: my completed chunk straight into the root's output shard.
    cb_wait_front({CB_SENDG}, {geo.a});
    cb_wait_front({CB_SENDU}, {geo.a});
    noc_async_write(get_read_ptr({CB_SENDG}), get_noc_addr(root_vx, root_vy, out_addr + offset * {pb}),
                    {geo.a * pb});
    noc_async_write(get_read_ptr({CB_SENDU}),
                    get_noc_addr(root_vx, root_vy, out_addr + ({geo.t} + offset) * {pb}), {geo.a * pb});
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_vx, root_vy, static_cast<uint32_t>(get_semaphore({SEM_GATHER}))), 1);
    noc_async_atomic_barrier();
    cb_pop_front({CB_SENDG}, {geo.a});
    cb_pop_front({CB_SENDU}, {geo.a});
"""
    else:  # tree / direct
        # A non-root node ships one WHOLE T-tile block per operand to its parent, into the parent's
        # landing slot `my_slot`; the source is cb_sendg/cb_sendu (a leaf just forwards its partial,
        # which compute copies there, so the writer has ONE source in every case).
        body = f"""
    const uint32_t parent_vx = get_arg_val<uint32_t>(7);
    const uint32_t parent_vy = get_arg_val<uint32_t>(8);
    const uint32_t my_slot = get_arg_val<uint32_t>(9);
    if (is_root) {{
        return;
    }}
    volatile tt_l1_ptr uint32_t* invite_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore({SEM_INVITE})));
    noc_semaphore_wait_min(invite_ptr, 1);
    cb_wait_front({CB_SENDG}, {geo.t});
    cb_wait_front({CB_SENDU}, {geo.t});
    noc_async_write(get_read_ptr({CB_SENDG}),
                    get_noc_addr(parent_vx, parent_vy, get_write_ptr({land_g} + my_slot)), {geo.t * pb});
    noc_async_write(get_read_ptr({CB_SENDU}),
                    get_noc_addr(parent_vx, parent_vy, get_write_ptr({land_u} + my_slot)), {geo.t * pb});
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(parent_vx, parent_vy,
                                   static_cast<uint32_t>(get_semaphore({SEM_DATA0} + my_slot))), 1);
    noc_async_atomic_barrier();
    cb_pop_front({CB_SENDG}, {geo.t});
    cb_pop_front({CB_SENDU}, {geo.t});
"""
    return _DF_INCLUDES + "\nvoid kernel_main() {\n" + head + body + "\n}\n"


# ---- COMPUTE ----------------------------------------------------------------

_COMPUTE_INCLUDES = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

using namespace compute_kernel_lib;

// The op's own blocked-eltwise spelling. NOT optional: bare `input(cb)` / `output(cb)` default to
// per-TILE lifecycles, which makes eltwise_chain silently clamp block_size to 1.
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
"""


def _matmul_call(geo, in1_cb, out_cb):
    osw = _largest_divisor_le(geo.n, DEST_LIMIT_TILES)
    in1_sb = geo.n // osw
    return f"""
    matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndRetainOnLastBlock,   // x is resident and reused by BOTH matmuls
        InputPolicy::WaitAndPopPerKBlock,
        NoPostCompute, NoPreKBlock, NoPostKBlock,
        /*untilize_block_ct_dim=*/0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset,
        /*caller_owns_pack_target=*/false>(
        x_buf, {in1_cb}, {out_cb}, {out_cb},
        MatmulBlockShape::of(/*in0_num_subblocks=*/{geo.m}, /*in1_num_subblocks=*/{in1_sb},
                             /*out_subblock_h=*/1, /*out_subblock_w=*/{osw},
                             /*in0_block_k=*/{geo.kr}, /*num_k_blocks=*/1),
        {{}}, {{}}, /*in1_per_core_w=*/{geo.n}, /*out_row_width=*/{geo.n});
"""


_UP = "WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block"
_ACC_PACK = (
    "PackTile<output({acc}, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled,"
    " PackRelu::Disabled, L1Accumulation::Disabled, DestAccumulation::Disabled, TileOffset::Set)>"
)


def _reduce_code(mech, nc, n, land_base, acc_cb, out_cb, indent="    "):
    """The accumulate mechanism, emitted for `nc` contributors of `n` tiles each landing in
    consecutive CBs [land_base, land_base + nc). The result ALWAYS lands in `out_cb`, a FRESH CB, so
    a dataflow consumer can never observe a mid-chain state of the accumulator."""
    L = [land_base + i for i in range(nc)]
    out = []
    if nc == 1:
        return indent + f"copy<blk_in({L[0]}), blk_out({out_cb})>({_ew(n)});\n"
    if mech == "addchain":
        out.append(f"copy<blk_in({L[0]}), blk_out({acc_cb})>({_ew(n)});")
        for i in range(1, nc - 1):
            out.append(f"add<blk_in({acc_cb}), blk_in({L[i]}), blk_out({acc_cb})>" f"({_ew(n)});")
        out.append(f"add<blk_in({acc_cb}), blk_in({L[nc - 1]}), blk_out({out_cb})>" f"({_ew(n)});")
    elif mech == "pack_l1_acc":
        pk = _ACC_PACK.format(acc=acc_cb)
        out.append("{")
        out.append(f"    CircularBuffer acc_buf({acc_cb});")
        out.append(f"    acc_buf.reserve_back({n});")
        out.append(f"    using AccFoldPack = {pk};")
        out.append(f"    eltwise_chain(EltwiseShape::tiles({n}), CopyTile<input({L[0]})>{{}}, AccFoldPack{{0}});")
        out.append("    pack_reconfig_l1_acc(1);")
        for i in range(1, nc):
            out.append(f"    eltwise_chain(EltwiseShape::tiles({n}), CopyTile<input({L[i]})>{{}}, AccFoldPack{{0}});")
        out.append("    pack_reconfig_l1_acc(0);")
        out.append(f"    acc_buf.push_back({n});")
        out.append("}")
        # The bf16 -> bfp8 conversion the mechanism owes the transport. Charged, not hidden.
        out.append(f"copy<blk_in({acc_cb}), blk_out({out_cb})>({_ew(n)});")
    elif mech == "dest_acc":
        blk = _largest_divisor_le(n, DEST_LIMIT_TILES)
        elems = [f"CopyTile<input({L[0]}, {_UP})>{{}}"]
        elems += [
            f"DestReuseBinary<input({c}, {_UP}), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{{}}" for c in L[1:]
        ]
        elems.append(f"PackTile<output({out_cb}, ReservePolicy::Upfront, PushPolicy::AtEnd)>{{}}")
        joined = (",\n" + indent + "    ").join(elems)
        out.append(f"eltwise_chain(EltwiseShape::tiles({n}, {blk}),\n{indent}    {joined});")
    elif mech == "pack_l1_pair":
        blk = _largest_divisor_le(n, DEST_LIMIT_TILES)
        pk = _ACC_PACK.format(acc=acc_cb)
        out.append("{")
        out.append(f"    CircularBuffer acc_buf({acc_cb});")
        out.append(f"    acc_buf.reserve_back({n});")
        out.append(f"    using AccFoldPack = {pk};")
        first = True
        for i in range(0, nc - 1, 2):
            out.append(
                f"    eltwise_chain(EltwiseShape::tiles({n}, {blk}),\n"
                f"        BinaryFpu<input({L[i]}, {_UP}), input({L[i + 1]}, {_UP}), BinaryFpuOp::Add>{{}},\n"
                f"        AccFoldPack{{0}});"
            )
            if first:
                out.append("    pack_reconfig_l1_acc(1);")
                first = False
        if nc % 2:
            out.append(
                f"    eltwise_chain(EltwiseShape::tiles({n}, {blk}), "
                f"CopyTile<input({L[nc - 1]}, {_UP})>{{}}, AccFoldPack{{0}});"
            )
        out.append("    pack_reconfig_l1_acc(0);")
        out.append(f"    acc_buf.push_back({n});")
        out.append("}")
        out.append(f"copy<blk_in({acc_cb}), blk_out({out_cb})>({_ew(n)});")
    else:
        raise ValueError(f"unknown mech {mech!r}")
    return "".join(indent + line + "\n" for line in out)


def _compute_source(shape, geo, p, mech, slots):
    """RT args: 0 my_row, 1 is_root, 2 assigned, 3 n_peer."""
    land_g, land_u = CB_LAND_G, CB_LAND_G + max(p["n_slots"], 1)
    head = f"""
void kernel_main() {{
    const uint32_t my_row = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t assigned = get_arg_val<uint32_t>(2);
    const uint32_t n_peer = get_arg_val<uint32_t>(3);
    (void)my_row; (void)is_root; (void)assigned; (void)n_peer;

    compute_kernel_hw_startup({CB_X}, {CB_WG}, {CB_PG});
    CircularBuffer x_buf({CB_X}), wg_buf({CB_WG}), wu_buf({CB_WU}), pg_buf({CB_PG}), pu_buf({CB_PU});
{_matmul_call(geo, "wg_buf", "pg_buf")}
{_matmul_call(geo, "wu_buf", "pu_buf")}
"""
    if shape == "mm_only":
        body = ""
    elif shape in ("scatter", "scatter_dual"):
        body = f"""
    if (assigned == 0) {{
        return;  // fewer uniform slices than cores: this core still CONTRIBUTES, it just owns none
    }}
{_reduce_code(mech, geo.k, geo.a, land_g, CB_ACCG, CB_SENDG, indent="    ")}
{_reduce_code(mech, geo.k, geo.a, land_u, CB_ACCU, CB_SENDU, indent="    ")}
"""
    elif shape == "ring":
        rot = f"""
    // CB ROTATION, zero copies. A CB is circular and cb_pg holds EXACTLY T pages, so popping
    // my_row*a pages and immediately re-reserving+re-pushing the same count (without writing
    // anything) leaves those physical pages — chunks 0..my_row-1 — at the END of the FIFO. The
    // walk that follows therefore visits chunks my_row, my_row+1, ..., wrapping to 0..my_row-1,
    // which is EXACTLY the ring's own-chunk order. The alternative (staging the rotation through
    // a scratch CB) would cost a full T-tile local L1 copy per operand, i.e. it would price the
    // ring's chunk selection into the ring's own measurement.
    if (my_row) {{
        const uint32_t rot = my_row * {geo.a};
        pg_buf.wait_front(rot); pg_buf.pop_front(rot);
        pg_buf.reserve_back(rot); pg_buf.push_back(rot);
        pu_buf.wait_front(rot); pu_buf.pop_front(rot);
        pu_buf.reserve_back(rot); pu_buf.push_back(rot);
    }}
"""
        steps = [
            f"    copy<blk_in({CB_PG}), blk_out({CB_SENDG})>({_ew(geo.a)});",
            f"    copy<blk_in({CB_PU}), blk_out({CB_SENDU})>({_ew(geo.a)});",
        ]
        for s in range(geo.k - 1):
            steps.append(f"    add<blk_in({land_g + s}), blk_in({CB_PG}), blk_out({CB_SENDG})>" f"({_ew(geo.a)});")
            steps.append(f"    add<blk_in({land_u + s}), blk_in({CB_PU}), blk_out({CB_SENDU})>" f"({_ew(geo.a)});")
        body = rot + "\n".join(steps) + "\n"
    else:  # tree / direct
        # A node folds `n_peer` landed blocks onto its own partial. The landed blocks arrive in
        # `slots` CBs round-robin, so the fold is a runtime loop with a compile-time slot switch.
        #
        # GATE AND UP ARE FOLDED IN THE SAME ITERATION, not in two passes. Two passes DEADLOCK with
        # the reader's slot recycling: the reader may only invite contributor c+slots after it has
        # re-reserved BOTH landing CBs of that slot, and a separate up-pass would not pop
        # land_u[slot] until the whole gate pass had finished — which itself needs that invite.
        ew = _ew(geo.t)

        def pair(slot, src_g, src_u, dst_g, dst_u):
            return (
                f"add<blk_in({src_g}), blk_in({land_g + slot}), blk_out({dst_g})>({ew}); "
                f"add<blk_in({src_u}), blk_in({land_u + slot}), blk_out({dst_u})>({ew});"
            )

        def switch(src_g, src_u, dst_g, dst_u):
            return "\n".join(
                f"                case {s}: {pair(s, src_g, src_u, dst_g, dst_u)} break;" for s in range(slots)
            )

        sw_only_r = switch(CB_PG, CB_PU, CB_OUT, CB_OUT)
        sw_only_s = switch(CB_PG, CB_PU, CB_SENDG, CB_SENDU)
        sw_first = switch(CB_PG, CB_PU, CB_ACCG, CB_ACCU)
        sw_mid = switch(CB_ACCG, CB_ACCU, CB_ACCG, CB_ACCU)
        sw_last_r = switch(CB_ACCG, CB_ACCU, CB_OUT, CB_OUT)
        sw_last_s = switch(CB_ACCG, CB_ACCU, CB_SENDG, CB_SENDU)
        body = f"""
    if (n_peer == 0) {{
        // Leaf: forward my own partial untouched.
        if (is_root) {{
            copy<blk_in({CB_PG}), blk_out({CB_OUT})>({ew});
            copy<blk_in({CB_PU}), blk_out({CB_OUT})>({ew});
        }} else {{
            copy<blk_in({CB_PG}), blk_out({CB_SENDG})>({ew});
            copy<blk_in({CB_PU}), blk_out({CB_SENDU})>({ew});
        }}
        return;
    }}
    for (uint32_t c = 0; c < n_peer; ++c) {{
        const uint32_t slot = c % {slots};
        const bool last = (c + 1 == n_peer);
        if (c == 0 && last) {{
            if (is_root) {{
                switch (slot) {{
{sw_only_r}
                }}
            }} else {{
                switch (slot) {{
{sw_only_s}
                }}
            }}
        }} else if (c == 0) {{
            switch (slot) {{
{sw_first}
            }}
        }} else if (!last) {{
            switch (slot) {{
{sw_mid}
            }}
        }} else if (is_root) {{
            switch (slot) {{
{sw_last_r}
            }}
        }} else {{
            switch (slot) {{
{sw_last_s}
            }}
        }}
    }}
"""
    return _COMPUTE_INCLUDES + head + body + "\n}\n"


# ===========================================================================
# Program descriptor
# ===========================================================================


def create_descriptor(device, x_t, wg_t, wu_t, out_t, shape, geo, mech="addchain", slots=1, root=0, fidelity=None):
    ok, why = feasible(shape, geo, mech, slots)
    if not ok:
        raise ValueError(f"{shape}/{mech} at {geo}: {why}")
    p = plan(shape, geo, slots)
    cr = geo.core_range
    n_slots = max(p["n_slots"], 1)
    land_g, land_u = CB_LAND_G, CB_LAND_G + n_slots

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_t),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WG, wg_t),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WU, wu_t),
        _cb(CB_PG, cr, geo.t),
        _cb(CB_PU, cr, geo.t),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_t),
    ]
    if shape != "mm_only":
        acc_dtype = ttnn.bfloat16 if mech in _BF16_MECHS else ttnn.bfloat8_b
        acc_bytes = _acc_bytes(mech)
        if shape in ("scatter", "scatter_dual"):
            send_pages, acc_pages = geo.a, geo.a
        elif shape == "ring":
            send_pages, acc_pages = 2 * geo.a, 0
        else:
            send_pages, acc_pages = geo.t, geo.t
        cbs += [_cb(CB_SENDG, cr, send_pages), _cb(CB_SENDU, cr, send_pages)]
        if acc_pages and mech != "dest_acc":
            cbs += [
                _cb(CB_ACCG, cr, acc_pages, acc_bytes, acc_dtype),
                _cb(CB_ACCU, cr, acc_pages, acc_bytes, acc_dtype),
            ]
        for i in range(p["n_slots"]):
            cbs.append(_cb(land_g + i, cr, p["slot_tiles"]))
            cbs.append(_cb(land_u + i, cr, p["slot_tiles"]))

    out_addr = out_t.buffer_address()
    tree = p["tree"]
    # Per-core runtime args, laid out identically for the reader / writer / compute triple.
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for col in range(geo.ncols):
        rvx, rvy = _virtual(device, col, root)
        for row in range(geo.k):
            is_root = 1 if row == root else 0
            assigned = p["assigned"][row]
            offset = p["offsets"][row]
            if shape == "mm_only":
                peers, dests, n_peer, my_slot, pvx, pvy = [], [], 0, 0, 0, 0
            elif shape in ("scatter", "scatter_dual"):
                peers = [_virtual(device, col, r) for r in range(geo.k)]
                dests = [(*_virtual(device, col, r), p["offsets"][r], p["assigned"][r]) for r in range(geo.k)]
                n_peer, my_slot, pvx, pvy = geo.k, row, 0, 0
            elif shape == "ring":
                right = (row + 1) % geo.k
                left = (row - 1) % geo.k
                peers = [_virtual(device, col, right)]
                dests = []
                n_peer, my_slot = geo.k - 1, 0
                pvx, pvy = _virtual(device, col, left)
            elif shape == "direct":
                kids = [r for r in range(geo.k) if r != root] if is_root else []
                peers = [_virtual(device, col, r) for r in kids]
                dests = []
                n_peer = len(kids)
                my_slot = 0 if is_root else ([r for r in range(geo.k) if r != root].index(row) % slots)
                pvx, pvy = rvx, rvy
            else:  # tree
                kids = tree[row]["children"]
                peers = [_virtual(device, col, r) for r in kids]
                dests = []
                n_peer = len(kids)
                par = tree[row]["parent"]
                pvx, pvy = _virtual(device, col, par) if par is not None else (rvx, rvy)
                my_slot = (tree[par]["children"].index(row) % slots) if par is not None else 0
                assigned = geo.t

            r_args = [row, is_root, assigned, offset, rvx, rvy, out_addr, n_peer]
            for vx, vy in peers:
                r_args += [vx, vy]
            if shape == "scatter_dual":
                for vx, vy, doff, da in dests:
                    r_args += [vx, vy, doff, da]
            reader_rt[col][row] = r_args

            w_args = [row, is_root, assigned, offset, rvx, rvy, out_addr, pvx, pvy, my_slot]
            for vx, vy, doff, da in dests:
                w_args += [vx, vy, doff, da]
            writer_rt[col][row] = w_args
            compute_rt[col][row] = [row, is_root, assigned, n_peer]

    kernels = [
        _kernel(_reader_source(shape, geo, p, mech, slots), cr, reader_rt, ttnn.ReaderConfigDescriptor()),
        _kernel(_writer_source(shape, geo, p, mech, slots), cr, writer_rt, ttnn.WriterConfigDescriptor()),
        _kernel(_compute_source(shape, geo, p, mech, slots), cr, compute_rt, compute_config(fidelity)),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cr, initial_value=0) for i in range(NUM_SEMAPHORES)]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def run(device, x_t, wg_t, wu_t, out_t, shape, geo, mech="addchain", slots=1, fidelity=None):
    desc = create_descriptor(device, x_t, wg_t, wu_t, out_t, shape, geo, mech, slots, fidelity=fidelity)
    return ttnn.generic_op([x_t, wg_t, wu_t, out_t], desc)
