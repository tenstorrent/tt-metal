# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: make the rms_norm cross-core combine a 2-DEEP PIPELINE.

NOT the op.  The concept under test is a SCHEDULE, not a kernel primitive:
whether block r+1's pass A (square + reduce -- entirely local, no cross-core
dependency) can be in flight while block r's combine round is still gathering,
folding and multicasting.  A schedule can only be measured against the work it
is supposed to overlap, so the bench keeps the op's real geometry and real
per-stage costs and changes ONLY the issue order plus the buffer depths the new
order needs.

How the isolation works
-----------------------
The bench asks the OP'S OWN descriptor for the program (so BLOCK_ROWS, the L1
solve, the mcast wiring, the shard-backed CBs and every CT/RT arg are exactly
the op's), then rewrites three things on the returned ``ttnn.ProgramDescriptor``:

  * the three ``kernel_source`` paths -> ``kernels/bench_*.cpp``, which are
    VERBATIM copies of the op's kernels plus an ``RMS_PIPE_VARIANT`` switch;
  * ``defines["RMS_PIPE_VARIANT"]`` -> the variant number;
  * the two combine CBs' ``total_size`` for the variants that need depth.

Variant 0 is therefore the op's current approach byte-for-byte -- the honest
baseline -- and every candidate differs from it by the switch alone.

Variants -- a BITMASK, so every lever is measurable alone and in composition
---------------------------------------------------------------------------
0   ``serial``   the op today: per row-block, ship -> wait for GROUP_SIZE-1
                 arrivals -> fold -> finalize -> multicast -> pass B.
1   ``mcast``    writer, ZERO extra L1: the root publishes its own finalized stat
                 to cb_row_final BEFORE the multicast, so its pass B does not wait
                 out the broadcast.
2   ``pipe``     compute, ZERO extra L1: pass A runs ONE ROW-BLOCK AHEAD on every
                 core, so the root's arrival wait and the whole root chain overlap
                 independent square+reduce work.
4   ``ring2``    the gather ring 2 ROUNDS deep with round-parity halves.  Measured
                 UNNECESSARY (the writer's own happens-before chain already keeps a
                 member exactly one round behind) and a measured REGRESSION.
8   ``hand2``    cb_sum_handoff 2 row-blocks deep, so pass A one block ahead never
                 stalls on the writer still owning the last partial.
16  ``xdeep``    a READER-FED cb_input_tiles sized to num_blocks + 1 row-blocks --
                 required for CORRECTNESS of ``pipe`` off the shard-backed path.

Precision contract is FIXED and identical for every variant (it comes from the
caller's ComputeConfigDescriptor, untouched).  Every variant that is correct is
BIT-IDENTICAL to ``serial``: the levers change WHEN work is issued, never what.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path


import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd


# `torch` is imported LAZILY here, on first attribute access, rather than at module scope.
# `ttnn/ttnn/operations/__init__.py` runs pkgutil.walk_packages over the operations tree, and
# the repo's `check-torch-imports-in-ttnn` pre-commit hook forbids a global torch import
# anywhere under ttnn/ for exactly that reason.  See perf_experiments/README.md -- an
# __init__.py in that directory once broke `import ttnn` repo-wide twice in one round.
# Every `torch.<attr>` use below is unchanged; the proxy just defers the import.
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INPUT_TILES = 1
CB_SUM_HANDOFF = 10
CB_PARTIALS_GATHERED = 11

# The variant number is a BITMASK over four independent levers:
MCAST_EARLY = 1  # writer:  root publishes its own stat BEFORE the broadcast
PIPE_A = 2  # compute: pass A runs one row-block AHEAD
RING2 = 4  # writer+compute: 2-round gather ring, round-parity halves
HANDOFF2 = 8  # host:    cb_sum_handoff 2 row-blocks deep
# host-only: size a READER-FED cb_input_tiles to num_blocks + 1 row-blocks.
#
# Pass A one block ahead reads a TWO-ROW-BLOCK window of cb_input_tiles at a tile
# OFFSET from the CB front, and a tile offset cannot cross the ring wrap.  When the
# CB is backed on the resident input shard (every SHARDED combine plan) the ring IS
# the whole per-core assignment, so the front never wraps and the window is always
# contiguous.  On the INTERLEAVED width split the CB is a reader-fed ring of
# CB_X_DEPTH (== 2) row-blocks, and the window straddles the wrap on every round
# whose block index is N-1 mod N -- measured as pcc 0.9802 / not-bit-exact.  This
# bit is the FIX, and its price: the ring must hold num_blocks + 1 row-blocks, i.e.
# effectively the whole assignment, which is what the shard-backed CB gives free.
XDEEP = 16

_BITNAMES = (
    (MCAST_EARLY, "mcast"),
    (PIPE_A, "pipe"),
    (RING2, "ring2"),
    (HANDOFF2, "hand2"),
    (XDEEP, "xdeep"),
)


def variant_name(v):
    if v == 0:
        return "serial"
    return "+".join(n for b, n in _BITNAMES if v & b)


class VARIANTS(dict):
    def __missing__(self, v):  # any bitmask is a legal variant
        return variant_name(v)


VARIANTS = VARIANTS()


def _depth(variant):
    """(cb_sum_handoff multiplier, cb_partials_gathered multiplier)."""
    return (2 if variant & HANDOFF2 else 1, 2 if variant & RING2 else 1)


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
    """The op's program descriptor, repointed at the bench kernels + variant."""
    cfg = compute_config or _perf_config()
    pd = rpd.create_program_descriptor(x, out, gamma=gamma, epsilon=epsilon, compute_kernel_config=cfg)

    names = ["bench_reader.cpp", "bench_writer.cpp", "bench_compute.cpp"]
    for k, name in zip(pd.kernels, names):
        k.kernel_source = str(KERNEL_DIR / name)
        k.defines = [("RMS_PIPE_VARIANT", str(variant))]

    # BLOCK_ROWS / GROUP_SIZE straight off the compute kernel's CT args, so the
    # bench never restates a knob the descriptor owns.
    ct = list(pd.kernels[2].compile_time_args)
    block_rows, group_size, combine = ct[3], ct[13], ct[12]
    assert combine == 1, "this bench only measures the COMBINE path"

    sum_mult, gath_mult = _depth(variant)
    extra_bytes = 0
    if sum_mult != 1:
        cb, fmt = _cb_by_index(pd, CB_SUM_HANDOFF)
        add = (sum_mult - 1) * block_rows * fmt.page_size
        cb.total_size += add
        extra_bytes += add
    if gath_mult != 1:
        cb, fmt = _cb_by_index(pd, CB_PARTIALS_GATHERED)
        add = (gath_mult - 1) * group_size * block_rows * fmt.page_size
        cb.total_size += add
        extra_bytes += add

    # rows-per-core off the compute kernel's OWN runtime args (arg 0 == row_count),
    # so num_blocks is the plan's, never re-derived from the shard shape (which the
    # interleaved width split does not have).
    # (RuntimeArgsColProxy exposes __getitem__ but no __len__, hence the probe loop.)
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
    native_in = list(pd.kernels[0].compile_time_args)[13]
    if variant & XDEEP and not native_in:
        wt_chunk = ct[1]
        cb, fmt = _cb_by_index(pd, CB_INPUT_TILES)
        want = (num_blocks + 1) * block_rows * wt_chunk * fmt.page_size
        if want > cb.total_size:
            extra_bytes += want - cb.total_size
            cb.total_size = want

    info = dict(
        block_rows=block_rows,
        group_size=group_size,
        extra_l1_bytes=extra_bytes,
        rows_per_core=max_rows,
        num_blocks=num_blocks,
        native_in=int(native_in),
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


def device_kernel_ns(op_code_contains="GenericOp"):
    """The newest ops_perf_results CSV's DEVICE KERNEL DURATION [ns]."""
    reports = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
    assert reports, "no profiler report found -- run under scripts/run_safe_pytest.sh --profile"
    out = []
    with open(reports[-1]) as f:
        for row in csv.DictReader(f):
            ns = row.get("DEVICE KERNEL DURATION [ns]")
            if ns and ns.strip():
                out.append((row.get("OP CODE", "").strip(), int(float(ns))))
    return reports[-1], out
