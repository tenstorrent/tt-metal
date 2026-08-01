# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ROUND-2 isolated bake-off for moe_fused_swiglu's ROOT reduce-accumulate mechanism.

See bench.py's module docstring for the menu, the round-1 delta (the baseline is now the SHIPPED
`ELTWISE_BLK = 8` spelling, not round 1's silently-clamped one) and the two load-bearing kernel
notes.

Correctness gate: device output read back as float32 vs a full-fp32 torch reference
(local_partial + repeats * sum(children), no intermediate quantization) — the SAME reference for
every variant, so PCC directly prices each mechanism's own requantization cost. The op's soft gate
is 0.975 against a measured bfp4 format floor of 0.9797-0.9802, so ANY option that loses PCC here
is a regression, not a trade.

Run:
  scripts/run_safe_pytest.sh --run-all \\
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/root_accum_mechanism_r2/test_root_accum_mechanism_r2.py \\
    -k menu_focus
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
from pathlib import Path

import pytest
from loguru import logger

# NOTE: `torch` is imported LAZILY at every use site. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/`.
import ttnn

from models.common.utility_functions import comp_pcc

# Loaded by explicit file path so this experiment stays self-contained and never depends on an
# __init__.py in the shared perf_experiments/ parent (parallel sibling part-optimizers own their own
# dirs there).
_MOD_PATH = Path(__file__).resolve().parent / "bench.py"
_spec = importlib.util.spec_from_file_location("root_accum_mechanism_r2_bench", _MOD_PATH)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

V = _bench
NAMES = _bench.VARIANT_NAMES

PCC_THRESHOLD = 0.975  # the op's soft gate (feature_spec extras)
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# --- focus shape: emb 7168, capacity 5120, count 256 -> HGROUPS 11, HN_PAD 6, M_BLOCK 8, m_eff 8,
#     gu_block_tiles = 48 bfp8 tiles, root fan-in 4, ELTWISE_BLK 8. Two roles = gate + up.
FOCUS_FAN_IN = 4
FOCUS_TILES = 48
ELTWISE_BLK = 8
ROLES = 2

MENU = (
    V.VARIANT_BASELINE,
    V.VARIANT_PERTILE,
    V.VARIANT_PACK_L1_ACC,
    V.VARIANT_PACK_L1_BFP8,
    V.VARIANT_DEST_PAIR,
    V.VARIANT_DEST_FULL,
    V.VARIANT_PACK_L1_FULL,
    V.VARIANT_PINGPONG,
    V.VARIANT_PACK_L1_PAIR,
    V.VARIANT_PACK_L1_PAIR_ODDADD,
)

# Predicate sweep: m_eff 1/2/4/8 -> 6/12/24/48 tiles at HN_PAD 6, x the real tree's fan-ins.
SWEEP_TILES = (6, 12, 24, 48)
SWEEP_FAN_INS = (1, 2, 3, 4)
SWEEP_VARIANTS = (
    V.VARIANT_BASELINE,
    V.VARIANT_PACK_L1_PAIR,
    V.VARIANT_PACK_L1_PAIR_ODDADD,
    V.VARIANT_PACK_L1_FULL,
)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure(device, *, variant, fan_in, block_tiles, repeats, roles=ROLES, seed_val=0):
    """One fresh-cache run of one variant (perf-measure discipline: device kernel time has no
    warm-up transient). Returns (ns, [pcc per role])."""
    import torch

    tensors, src = V.make_operands(
        device,
        fan_in=fan_in,
        block_tiles=block_tiles,
        acc_dtype=V.ACC_DTYPE[variant],
        roles=roles,
        variant=variant,
        seed_val=seed_val,
    )
    try:
        V.run(
            tensors,
            variant=variant,
            fan_in=fan_in,
            block_tiles=block_tiles,
            blk=ELTWISE_BLK,
            repeats=repeats,
            roles=roles,
        )
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        pccs = []
        for r in range(roles):
            got = ttnn.to_torch(V.acc_tensor(tensors, r)).to(torch.float32)
            ref = V.expected(src, r, repeats)
            _, pcc = comp_pcc(ref, got, PCC_THRESHOLD)
            pccs.append(float(pcc))
    finally:
        V.free(tensors)
    return ns, pccs


def test_menu_focus(device):
    """THE DELIVERABLE: every mechanism on the focus shape (fan_in 4, 48 tiles, gate+up), with
    device ns at REPEATS=1 and REPEATS=9, PCC, and the L1 bytes each option costs.

    The two REPEATS points difference out the constant launch/teardown cost:
    per-M-block cost = (T(9) - T(1)) / 8. T(1) is what the op pays at count 256 (m_blocks = 1);
    the slope is the mechanism cost with launch overhead removed."""
    rows = []
    for variant in MENU:
        ns1, pcc1 = _measure(device, variant=variant, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_TILES, repeats=1)
        ns9, pcc9 = _measure(device, variant=variant, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_TILES, repeats=9)
        assert ns1 is not None and ns9 is not None, f"profiler produced no data for {NAMES[variant]}"
        rows.append(
            {
                "variant": variant,
                "ns1": ns1,
                "ns9": ns9,
                "slope": (ns9 - ns1) / 8.0,
                "pcc": min(pcc1),
                "pcc9": min(pcc9),
                "acc_dtype": str(V.ACC_DTYPE[variant]).replace("DataType.", ""),
                "slots": V.reduce_slots_needed(variant, FOCUS_FAN_IN),
                "l1": V.l1_delta_bytes(variant, FOCUS_FAN_IN),
            }
        )

    base = next(r for r in rows if r["variant"] == V.VARIANT_BASELINE)
    lines = [
        "",
        "=== root_accum_mechanism_r2 MENU (focus: fan_in 4, 48 tiles, gate+up, ELTWISE_BLK 8) ===",
        f"{'variant':>17} {'acc_fmt':>10} {'slots':>5} {'+L1 B':>9} "
        f"{'ns@R1':>9} {'x':>6} {'ns/Mblk':>9} {'x':>6} {'pcc@R1':>9} {'pcc@R9':>9}",
    ]
    for r in rows:
        lines.append(
            f"{NAMES[r['variant']]:>17} {r['acc_dtype']:>10} {r['slots']:>5} {r['l1']:>9} "
            f"{r['ns1']:>9.0f} {base['ns1']/r['ns1']:>6.2f} {r['slope']:>9.0f} "
            f"{base['slope']/r['slope']:>6.2f} {r['pcc']:>9.5f} {r['pcc9']:>9.5f}"
        )
    logger.info("\n".join(lines))


def test_predicate_sweep(device):
    """Where does each win hold? m_eff 1/2/4/8 (6/12/24/48 tiles) x fan_in 1/2/3/4, gate+up,
    REPEATS=1 (what the op pays per M-block at count 256)."""
    results = {}
    for block_tiles in SWEEP_TILES:
        for fan_in in SWEEP_FAN_INS:
            for variant in SWEEP_VARIANTS:
                ns, pccs = _measure(device, variant=variant, fan_in=fan_in, block_tiles=block_tiles, repeats=1)
                assert ns is not None
                results[(block_tiles, fan_in, variant)] = (ns, min(pccs))

    lines = ["", "=== root_accum_mechanism_r2 PREDICATE SWEEP (ns, and x vs baseline) ==="]
    header = f"{'tiles':>6} {'fan_in':>6}"
    for variant in SWEEP_VARIANTS:
        header += f" {NAMES[variant]:>16}"
    lines.append(header)
    for block_tiles in SWEEP_TILES:
        for fan_in in SWEEP_FAN_INS:
            base = results[(block_tiles, fan_in, V.VARIANT_BASELINE)][0]
            row = f"{block_tiles:>6} {fan_in:>6}"
            for variant in SWEEP_VARIANTS:
                ns, _ = results[(block_tiles, fan_in, variant)]
                row += f" {ns:>8.0f}/{base/ns:>5.2f}x"
            lines.append(row)
    lines.append("")
    lines.append("min PCC per cell:")
    for block_tiles in SWEEP_TILES:
        for fan_in in SWEEP_FAN_INS:
            row = f"{block_tiles:>6} {fan_in:>6}"
            for variant in SWEEP_VARIANTS:
                _, pcc = results[(block_tiles, fan_in, variant)]
                row += f" {pcc:>16.5f}"
            lines.append(row)
    logger.info("\n".join(lines))


def test_single_role_costing(device):
    """The winner costs +196,608 B of L1 (bf16 accumulator 92,160 + REDUCE_SLOTS 2 -> 104,448),
    which is 53,248 B OVER the 143,360 B the coordinator has free. This prices the HALF option:
    convert only ONE of the two roles (e.g. `up`), for +98,304 B — in budget.

    ROLES=1 measurements let the mixed configuration be costed by arithmetic: the two roles' passes
    execute back to back on the same three TRISCs, so
    mixed ~= baseline(1 role) + pack_l1_pair(1 role) - launch_overhead, and launch_overhead is
    obtained from the same variant's REPEATS slope."""
    rows = []
    for variant in (V.VARIANT_BASELINE, V.VARIANT_PACK_L1_PAIR):
        for roles in (1, 2):
            ns1, _ = _measure(
                device, variant=variant, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_TILES, repeats=1, roles=roles
            )
            ns9, pcc9 = _measure(
                device, variant=variant, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_TILES, repeats=9, roles=roles
            )
            rows.append((variant, roles, ns1, ns9, (ns9 - ns1) / 8.0, min(pcc9)))

    by = {(v, r): (n1, n9, sl) for v, r, n1, n9, sl, _ in rows}
    overhead = by[(V.VARIANT_BASELINE, 2)][0] - by[(V.VARIANT_BASELINE, 2)][2]
    mixed_est = by[(V.VARIANT_BASELINE, 1)][2] + by[(V.VARIANT_PACK_L1_PAIR, 1)][2] + overhead

    lines = ["", "=== root_accum_mechanism_r2 SINGLE-ROLE COSTING (fan_in 4, 48 tiles) ==="]
    for variant, roles, ns1, ns9, slope, pcc in rows:
        lines.append(f"  {NAMES[variant]:>15} roles={roles}  ns@R1={ns1:>8.0f}  ns/Mblk={slope:>8.0f}  pcc={pcc:.5f}")
    lines.append(f"  launch overhead (baseline roles=2: ns@R1 - ns/Mblk) = {overhead:.0f} ns")
    lines.append(
        f"  MIXED estimate (gate baseline + up pack_l1_pair) = {mixed_est:.0f} ns  vs "
        f"baseline(2 roles) {by[(V.VARIANT_BASELINE, 2)][0]:.0f} ns  = "
        f"{by[(V.VARIANT_BASELINE, 2)][0]/mixed_est:.2f}x   (+98,304 B of L1, IN budget)"
    )
    logger.info("\n".join(lines))


def test_reduce_scatter_regime(device):
    """DOES THE MECHANISM COMPOSE WITH A REDUCE-SCATTER RESTRUCTURE? (Perf 1 idea #3, 2.50x, owned
    by another subagent.) That restructure inverts this stage's shape: instead of a root folding
    fan_in <= 4 children over the WHOLE 48-tile block, each of the KGROUPS = 10 column cores folds
    ALL 10 contributors over a 1/10 SLICE of it. So the regime becomes many passes over few tiles —
    exactly where a per-pass fixed cost would dominate and where the pass-count-reducing mechanisms
    (dest_full / pack_l1_full / dest_pair) should look BEST. Measured at fan_in 8 and 10 over 5 and
    6 tiles (48 tiles split ~10 ways), plus 48 tiles at fan_in 10 as the un-scattered contrast."""
    cells = [(5, 8), (5, 10), (6, 8), (6, 10), (12, 10)]
    results = {}
    for block_tiles, fan_in in cells:
        for variant in SWEEP_VARIANTS + (V.VARIANT_PINGPONG,):
            ns, pccs = _measure(device, variant=variant, fan_in=fan_in, block_tiles=block_tiles, repeats=1)
            assert ns is not None
            results[(block_tiles, fan_in, variant)] = (ns, min(pccs))

    variants = SWEEP_VARIANTS + (V.VARIANT_PINGPONG,)
    lines = ["", "=== root_accum_mechanism_r2 REDUCE-SCATTER REGIME (ns / x-vs-baseline / min pcc) ==="]
    header = f"{'tiles':>6} {'fan_in':>6}"
    for variant in variants:
        header += f" {NAMES[variant]:>24}"
    lines.append(header)
    for block_tiles, fan_in in cells:
        base = results[(block_tiles, fan_in, V.VARIANT_BASELINE)][0]
        row = f"{block_tiles:>6} {fan_in:>6}"
        for variant in variants:
            ns, pcc = results[(block_tiles, fan_in, variant)]
            row += f" {ns:>8.0f}/{base/ns:>5.2f}x/{pcc:>7.5f}"
        lines.append(row)
    logger.info("\n".join(lines))


@pytest.mark.parametrize("variant", MENU)
def test_correctness(device, variant):
    """Every mechanism computes local_partial + REPEATS * sum(children). `pack_l1_bfp8` is EXPECTED
    to fail this — it is the documented correctness bug (the packer's L1-accumulate register does a
    linear add, invalid on a shared-exponent block-float tile) and is xfail'd, not silently skipped,
    so a future kernel_lib change that fixes it shows up as an unexpected pass."""
    ns, pccs = _measure(device, variant=variant, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_TILES, repeats=3)
    logger.info(f"{NAMES[variant]}: ns={ns:.0f} pcc={pccs}")
    if variant == V.VARIANT_PACK_L1_BFP8:
        assert min(pccs) < PCC_THRESHOLD, (
            f"pack_l1_acc onto a bfp8_b accumulator UNEXPECTEDLY passed (pcc={min(pccs)}); "
            "round 1 measured 0.412 — re-check whether the packer path changed"
        )
        return
    assert min(pccs) >= PCC_THRESHOLD, f"{NAMES[variant]}: pcc={min(pccs)} < {PCC_THRESHOLD}"
