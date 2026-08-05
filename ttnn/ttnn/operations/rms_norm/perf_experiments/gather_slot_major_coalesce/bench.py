# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: ONE contiguous NoC write per member per gather round.

NOT the op.  The concept under test is the cross-core combine's GATHER LAYOUT, and
what it buys is TRANSACTION COUNT (a different objective on the same knob D13
optimized for BYTES).

The problem
-----------
At the focus geometry (GROUP_SIZE 8, BLOCK_ROWS 8, GATHER_FACES 2) a member ships its
row-block's partial as 8 tile-rows x 2 faces = **16 `noc_async_write`s of 1024 B**, i.e.
16 kB of traffic carrying 8 rows x 32 fp32 = 1 kB of actual information.  Measured
effective rate ~13 GB/s -- small-transfer bound, far under a Blackhole NoC link.  The
reason it is 16 and not 1 is the ROW-MAJOR landing layout (D16): page
`r * GATHER_SLOTS + my_slot`, so a SENDER's tiles are on a stride of GATHER_SLOTS.

The idea
--------
Go back to a SLOT-MAJOR layout -- page `my_slot * BLOCK_ROWS + r` -- so a sender's
`rows` tiles are a CONTIGUOUS run, and at GATHER_FACES == 4 the whole block ships in
exactly **ONE** `noc_async_write`.  GATHER_FACES == 4 also DELETES THE NEED for the
`writer_gather_zero` boot at even GROUP_SIZE (that boot exists only to define the faces
the gather never writes), which the op measures at 2462 ns / 7.1% of the focus shape.
So the extra bytes of a whole-tile gather are paid for twice over -- once in transaction
count, once in the deleted boot.

Why this was closed and why it is open again
--------------------------------------------
`rms_norm_writer.cpp` records slot-major as abandoned because it "put a row's partials on
a stride of `rows` -- a gapped window no chain walk can express".  That was true of the
PRE-D22 `eltwise_chain` fold.  Perf 2's D22 replaced the chain with a raw fold that takes
EXPLICIT tile indices (`add_tiles(cb, cb, base + p, base + GATHER_HALF + p, 0)`), and an
explicit index walks a stride exactly as easily as a run.  VERIFIED by reading the fold
before anything was built; the objection is STALE.

How the isolation works
-----------------------
Same technique as the sibling `combine_pipeline_depth` bench.  The bench asks the OP'S OWN
descriptor for the program (so BLOCK_ROWS, the L1 solve, the mcast wiring, the shard-backed
CBs and every CT/RT arg are exactly the op's), then changes exactly two things:

  * the three `kernel_source` paths -> `kernels/bench_*.cpp`, VERBATIM copies of the op's
    kernels plus an `RMS_GATHER_LAYOUT` switch (and the pre-existing
    `RMS_ABLATE_GATHER_ZERO` ablation switch, used only for attribution);
  * `rpd.GATHER_FACES` (a module constant the descriptor reads at build time), for the
    faces axis.

`layout 0 / faces 2` is therefore the op today byte-for-byte -- the honest baseline -- and
every candidate differs from it by the switch(es) alone.  L1 is UNTOUCHED by every point in
the menu: `cb_partials_gathered` is `GATHER_SLOTS * BLOCK_ROWS` fp32 pages either way (the
layout permutes pages, it does not add them) and GATHER_FACES addresses faces INSIDE a page.

Precision contract is FIXED and identical for every variant: bf16 / TILE / HiFi2 /
fp32_dest_acc_en=False / math_approx_mode=False, straight from the caller's
ComputeConfigDescriptor and never a lever.  Every correct variant must be BIT-EXACT against
the baseline -- the fold's operand pairs and their order are unchanged, only the pages they
live on move -- and the test gates exactly that.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd


# `torch` is imported LAZILY, on first attribute access, rather than at module scope.
# `ttnn/ttnn/operations/__init__.py` runs pkgutil.walk_packages over the operations tree and
# the repo's `check-torch-imports-in-ttnn` pre-commit hook forbids a global torch import
# anywhere under ttnn/ for exactly that reason.  See perf_experiments/README.md.
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


KERNEL_DIR = Path(__file__).parent / "kernels"

CB_PARTIALS_GATHERED = 11
FP32_TILE_BYTES = 4096

ROW_MAJOR = 0  # the op's landing layout: page r * GATHER_SLOTS + my_slot
SLOT_MAJOR = 1  # the candidate:          page my_slot * BLOCK_ROWS + r


class Variant:
    """One point of the menu: (gather layout) x (GATHER_FACES) x (boot-zero ablated?).

    `zero_ablated` is NOT a proposal -- it is the attribution knob.  Running the baseline
    with the `writer_gather_zero` payload stripped (every CB handshake and trip count
    intact) is what separates "the boot's ns" from "the coalescing's ns", so the report can
    state the two independently instead of quoting their sum.
    """

    __slots__ = ("layout", "faces", "zero_ablated", "boot_mode", "falsify", "name")

    def __init__(self, layout, faces, zero_ablated=False, boot_mode=0, falsify=False):
        self.layout = layout
        self.faces = faces
        self.zero_ablated = zero_ablated
        # 0 = the op's ROOT-LOCAL boot.  1 = DISTRIBUTED: every core zeroes the unshipped
        # faces of ITS OWN slot in the root's CB (the root keeps its own slot + the pad).
        # Same total bytes, 1/GROUP_SIZE per core, in parallel, off the root's critical
        # path -- and every face stays DEFINED, so unlike `zero_ablated` this is a real
        # proposal and not just an attribution knob.
        self.boot_mode = boot_mode
        # BENCH SELF-CHECK, never a proposal: give the WRITER slot-major and the COMPUTE
        # row-major.  Every variant of this bake-off came out bit-exact on the first run,
        # which is exactly what a silently-dead `#define` would also look like -- so the
        # bench carries a point that MUST be wrong.  If `falsify` passes the correctness
        # gate, the switch is not reaching the kernels and every other number is void.
        self.falsify = falsify
        self.name = (
            ("sm" if layout == SLOT_MAJOR else "rm")
            + f"_f{faces}"
            + ("_nozero" if zero_ablated else "")
            + ("_dboot" if boot_mode == 1 else "")
            + ("_FALSIFY" if falsify else "")
        )

    def __repr__(self):
        return f"Variant({self.name})"


# ---------------------------------------------------------------------------
# THE MENU.  {row-major, slot-major} x {FACES 2, 3, 4}, as the task asks, plus the
# baseline-with-the-boot-ablated attribution point.
#
# The two axes are deliberately BOTH swept, because they answer different questions:
#   * FACES alone (rm_f3 / rm_f4) is the BYTE axis D13 already measured -- re-measured here
#     because at FACES == 4 the boot disappears, which D13's byte model did not price.
#   * LAYOUT alone at FACES < 4 (sm_f2 / sm_f3) is the ADDRESSING-ONLY CONTROL: slot-major
#     ships the SAME NUMBER of transactions there (a tile's shipped faces are still
#     discontiguous), so any delta at those points is layout overhead, not coalescing.
#   * sm_f4 is THE CANDIDATE and the only point where the transaction count collapses.
# ---------------------------------------------------------------------------
BASELINE = Variant(ROW_MAJOR, 2)
CANDIDATE = Variant(SLOT_MAJOR, 4)

FOCUS_MENU = [
    BASELINE,  # rm_f2      the op today
    Variant(ROW_MAJOR, 2, zero_ablated=True),  # rm_f2_nozero  attribution only
    Variant(ROW_MAJOR, 2, boot_mode=1),  # rm_f2_dboot   the boot's SAFE fix
    Variant(SLOT_MAJOR, 2, boot_mode=1),  # sm_f2_dboot   ... composed with the layout
    Variant(ROW_MAJOR, 3),  # rm_f3      byte axis
    Variant(ROW_MAJOR, 4),  # rm_f4      byte axis + boot deleted, NOT coalesced
    Variant(SLOT_MAJOR, 2),  # sm_f2      addressing-only control
    Variant(SLOT_MAJOR, 3),  # sm_f3      addressing-only control
    CANDIDATE,  # sm_f4      THE CANDIDATE: one write per member per round
]

# The sweep runs the five points the DOMAIN question needs (each extra point is a full JIT
# build on every geometry).  `rm_f4` vs `sm_f4` is the load-bearing pair: identical bytes,
# BLOCK_ROWS transactions vs ONE, so the delta between them is the COALESCING and nothing
# else.  `rm_f2` -> `rm_f4` is the byte axis; `rm_f2_nozero` prices the boot per regime.
SWEEP_MENU = [
    BASELINE,
    Variant(ROW_MAJOR, 2, zero_ablated=True),
    Variant(ROW_MAJOR, 2, boot_mode=1),  # the boot's SAFE fix: distributed, still defined
    Variant(ROW_MAJOR, 3),
    Variant(ROW_MAJOR, 4),
    CANDIDATE,
]

# The boot-lever menu on its own (all faces-defined, so all graduatable).
BOOT_MENU = [
    BASELINE,
    Variant(ROW_MAJOR, 2, zero_ablated=True),  # the ns ceiling for deleting the boot
    Variant(ROW_MAJOR, 2, boot_mode=1),  # the SAFE way to bank it
    Variant(SLOT_MAJOR, 2, boot_mode=1),  # ... composed with the layout (should be flat)
]

# The bench's own self-check: writer slot-major, compute row-major.  MUST be wrong.
FALSIFY = Variant(SLOT_MAJOR, 4, falsify=True)


def _perf_config():
    """The focus case's pinned compute config -- NEVER a lever."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def _cb_by_index(pd, buffer_index):
    for cb in pd.cbs:
        for fmt in cb.format_descriptors:
            if fmt.buffer_index == buffer_index:
                return cb, fmt
    return None, None


def build(x, out, gamma, *, variant, epsilon=1e-6, compute_config=None):
    """The op's program descriptor, repointed at the bench kernels + this variant."""
    cfg = compute_config or _perf_config()

    # GATHER_FACES is a module constant the descriptor reads at build time (it is D13's own
    # A/B handle).  Restored in the finally so one variant can never leak into the next.
    faces_orig = rpd.GATHER_FACES
    rpd.GATHER_FACES = variant.faces
    try:
        pd = rpd.create_program_descriptor(x, out, gamma=gamma, epsilon=epsilon, compute_kernel_config=cfg)
    finally:
        rpd.GATHER_FACES = faces_orig

    defines = [("RMS_GATHER_LAYOUT", str(variant.layout)), ("RMS_BOOT_MODE", str(variant.boot_mode))]
    if variant.zero_ablated:
        defines.append(("RMS_ABLATE_GATHER_ZERO", "1"))
    names = ["bench_reader.cpp", "bench_writer.cpp", "bench_compute.cpp"]
    for i, (k, name) in enumerate(zip(pd.kernels, names)):
        k.kernel_source = str(KERNEL_DIR / name)
        d = list(defines)
        if variant.falsify and i == 2:  # compute kernel: deliberately DISAGREE with the writer
            d[0] = ("RMS_GATHER_LAYOUT", "0")
        k.defines = d

    # Knobs read off the kernels' OWN args, so the bench never restates something the
    # descriptor owns.
    ct = list(pd.kernels[2].compile_time_args)
    block_rows, group_size, combine = ct[3], ct[13], ct[12]
    assert combine == 1, "this bench only measures the COMBINE path"
    wct = list(pd.kernels[1].compile_time_args)
    assert wct[15] == variant.faces, f"writer CT 15 is {wct[15]}, expected GATHER_FACES {variant.faces}"

    # rows-per-core off the compute kernel's own runtime args (arg 0 == row_count), so
    # num_blocks is the PLAN's and never re-derived from a shard shape the interleaved
    # width split does not have.
    rt = pd.kernels[2].runtime_args
    max_rows = 0
    for xi in range(len(rt)):
        col = rt[xi]
        for yi in range(64):
            try:
                a = list(col[yi])
            except Exception:
                break
            if a:
                max_rows = max(max_rows, int(a[0]))
    num_blocks = -(-max_rows // block_rows) if block_rows else 0

    gather_slots = group_size + group_size % 2
    _, gfmt = _cb_by_index(pd, CB_PARTIALS_GATHERED)
    gather_cb_bytes = gather_slots * block_rows * gfmt.page_size

    # The transaction model this bench exists to move.  Per MEMBER per ROUND.
    if variant.layout == SLOT_MAJOR and variant.faces == 4:
        txns = 1
        txn_bytes = block_rows * FP32_TILE_BYTES
    else:
        per_row = 2 if variant.faces == 2 else 1
        txns = block_rows * per_row
        txn_bytes = FP32_TILE_BYTES if variant.faces == 4 else (variant.faces if variant.faces == 3 else 1) * 1024
    # Boot-zero bytes.  `zero_bytes` is the TOTAL across the group; `root_zero_bytes` is
    # what lands on the ROOT'S critical path, which is the number the distributed boot moves.
    zero_bytes = 0
    root_zero_bytes = 0
    if not variant.zero_ablated:
        pad = gather_slots - group_size
        per_slot = block_rows * (2 if variant.faces == 2 else 1) * 1024 if variant.faces < 4 else 0
        zero_bytes = group_size * per_slot + pad * block_rows * FP32_TILE_BYTES
        root_zero_bytes = (per_slot if variant.boot_mode == 0 else 0) * (group_size - 1) + per_slot
        root_zero_bytes += pad * block_rows * FP32_TILE_BYTES

    info = dict(
        block_rows=block_rows,
        group_size=group_size,
        gather_slots=gather_slots,
        num_blocks=num_blocks,
        rows_per_core=max_rows,
        gather_cb_bytes=gather_cb_bytes,
        gather_txns_per_member_round=txns,
        gather_txn_bytes=txn_bytes,
        gather_bytes_per_member_round=txns * txn_bytes,
        boot_zero_bytes=zero_bytes,
        root_boot_zero_bytes=root_zero_bytes,
        boot_mode=variant.boot_mode,
    )
    return pd, info


def run(x, out, gamma, *, variant, epsilon=1e-6, compute_config=None):
    pd, info = build(x, out, gamma, variant=variant, epsilon=epsilon, compute_config=compute_config)
    tensors = [x] + ([gamma] if gamma is not None else []) + [out]
    return ttnn.generic_op(tensors, pd), info


# ---------------------------------------------------------------------------
# reference + measurement
# ---------------------------------------------------------------------------


def torch_reference(x_t, g_t, eps):
    xf = x_t.to(torch.float32)
    denom = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * denom
    if g_t is not None:
        y = y * g_t.to(torch.float32).reshape(1, 1, 1, -1)
    return y


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel_rms(got, ref):
    """The metric that caught two rms_norm bugs pcc held at 0.9997."""
    g = got.flatten().to(torch.float64)
    r = ref.flatten().to(torch.float64)
    d = r.pow(2).mean().sqrt()
    if d == 0:
        return 0.0
    return float((g - r).pow(2).mean().sqrt() / d)


def device_kernel_ns():
    """The newest ops_perf_results CSV's DEVICE KERNEL DURATION [ns] rows, in order."""
    reports = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
    assert reports, "no profiler report found -- run under scripts/run_safe_pytest.sh --profile"
    out = []
    with open(reports[-1]) as f:
        for row in csv.DictReader(f):
            ns = row.get("DEVICE KERNEL DURATION [ns]")
            if ns and ns.strip():
                out.append((row.get("OP CODE", "").strip(), int(float(ns))))
    return reports[-1], out
