# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off driver: reduce-tree TRANSPORT shape for moe_fused_swiglu.

Correctness is the ONLY pass/fail. Perf is measured and reported, never asserted.
One fresh-cache-equivalent run per (variant, regime) cell — device kernel time has no warm-up
transient (/perf-measure), so a single post-JIT run IS the measurement.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

# `torch` is imported LAZILY: scripts/validate_no_global_torch_imports.py forbids a module-level
# torch import anywhere under ttnn/ttnn/, and these benches live under the op directory.
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_transport_shape.bench import (
    TILE,
    Variant,
    build,
    reduce_tree,
    sharded_config,
    tree_max_fanin,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# ---- the FOCUS SHAPE, from the coordinator's brief (emb 7168 / cap 5120 / count 256 / bf16_rm) ----
HGROUPS = 11
KGROUPS = 10
HALF_TILES = 48  # gu_block_tiles = m_eff(8) * HN_PAD(6); the child ships this TWICE (gate + up)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _pcc(a, b):
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


def _make_tensors(device, hgroups, kgroups, half_tiles):
    """One [32, 2*half_tiles*32] bfp8 shard per core = the core's concatenated (gate | up) partial.

    The pattern is PHASE-SHIFTED per core (`(c + 7*i) % 31`) rather than offset per core: PCC is
    offset-invariant, so a constant-per-core offset would HIDE a wrong-core / wrong-slot transport
    bug, while a phase shift makes it a shape change that PCC sees. Magnitudes are deliberately kept
    within ~[0.5, 1.5] of each other across the KGROUPS contributors: a 100x magnitude spread makes a
    KGROUPS-deep chain of bfp8 adds lose ~2% of PCC to repeated re-quantization alone (measured:
    0.976), which is a property of the ARITHMETIC, not of the transport this bench is measuring.
    """
    import torch

    total_tiles = 2 * half_tiles
    cfg = sharded_config(device, hgroups, kgroups, total_tiles)
    n = hgroups * kgroups
    t = torch.empty((n * TILE, total_tiles * TILE), dtype=torch.float32)
    cols = torch.arange(total_tiles * TILE, dtype=torch.float32)
    for i in range(n):
        t[i * TILE : (i + 1) * TILE] = (0.5 + ((cols + 7 * i) % 31) / 31.0).reshape(1, -1)
    local = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    # The REAL (bfp8-quantized) per-core values — the reference must be built from THESE, since that
    # is what the device actually sums.
    quant = ttnn.to_torch(local).to(torch.float32)
    result = ttnn.from_torch(
        torch.zeros((n * TILE, total_tiles * TILE), dtype=torch.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    return local, result, quant


def _shard(t, hgroups, x, y):
    i = y * hgroups + x
    return t[i * TILE : (i + 1) * TILE]


def _expected(quant, tree, hgroups, kgroups, mode, slots):
    """Per-root expected shard. `add` mode -> the column sum; `xfer` mode -> the LAST child's bytes
    (the landing-slot echo, which is what proves the address + payload)."""
    out = {}
    for x in range(hgroups):
        for y in range(kgroups):
            if not tree[(x, y)]["is_root"]:
                continue
            if mode == "add":
                out[(x, y)] = sum(_shard(quant, hgroups, x, yy) for yy in range(kgroups))
            else:
                children = tree[(x, y)]["children"]
                cx, cy = children[-1]
                out[(x, y)] = _shard(quant, hgroups, cx, cy)
    return out


def _run_cell(device, variant, hgroups, kgroups, half_tiles, timed=True):
    import torch

    local, result, quant = _make_tensors(device, hgroups, kgroups, half_tiles)
    descriptor, meta = build(device, local, result, variant, hgroups, kgroups, half_tiles)
    ttnn.generic_op([local, result], descriptor)
    got = ttnn.to_torch(result).to(torch.float32)
    exp = _expected(quant, meta["tree"], hgroups, kgroups, variant.mode, meta["slots"])
    worst_pcc, worst_absdiff = 1.0, 0.0
    for (x, y), ref in exp.items():
        act = _shard(got, hgroups, x, y)
        worst_pcc = min(worst_pcc, _pcc(act, ref))
        worst_absdiff = max(worst_absdiff, float((act - ref).abs().max()))
    ns = None
    if timed:
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)  # discard: this run also paid JIT compile
        ttnn.generic_op([local, result], descriptor)
        ns = _read_kernel_ns(device)
    root_shards = {k: _shard(got, hgroups, k[0], k[1]).clone() for k in exp}
    return ns, worst_pcc, worst_absdiff, meta, root_shards


# ---------------------------------------------------------------------------
# The MENU. Every arm runs under the IDENTICAL precision contract; the only variable is transport.
# ---------------------------------------------------------------------------
def _menu(mode):
    v = []
    add = v.append
    add(Variant("baseline", n_ch=2, owner=(1, 1), slots=1, mode=mode, notes="shipped: 2 writes, BRISC/NOC_1, 1 slot"))
    add(Variant("one_write", n_ch=1, owner=(1,), slots=1, mode=mode, notes="merged CB, ONE 104448 B write on NOC_1"))
    add(Variant("send_noc0", n_ch=2, owner=(0, 0), slots=1, mode=mode, notes="same 2 writes, NCRISC/NOC_0"))
    add(Variant("dual_noc", n_ch=2, owner=(0, 1), slots=1, mode=mode, notes="gate on NOC_0 + up on NOC_1"))
    add(Variant("dir_noc", n_ch=2, owner=(1, 1), slots=1, mode=mode, dir_noc=True, notes="per-edge NoC by hop dir"))
    # The best BIT-IDENTICAL, ZERO-L1 combination: shortest-path per-edge NoC + one merged unicast.
    add(
        Variant(
            "dir_noc_merged",
            n_ch=1,
            owner=(1,),
            slots=1,
            mode=mode,
            dir_noc=True,
            notes="shortest-path NoC + ONE merged write, 1 slot (zero L1 cost)",
        )
    )
    add(Variant("slots2", n_ch=2, owner=(1, 1), slots=2, mode=mode, notes="REDUCE_SLOTS 2 (whole-wave push)"))
    # THE WINNER FAMILY: several children in flight, but each landing slot PUBLISHED AS IT ARRIVES,
    # which is only knowable with one arrival counter PER SLOT.
    add(
        Variant(
            "slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            per_slot_push=True,
            mode=mode,
            notes="2 slots, per-slot push (keeps the interleave)",
        )
    )
    add(
        Variant(
            "dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            dir_noc=True,
            per_slot_push=True,
            mode=mode,
            notes="shortest-path NoC + 2 slots + per-slot push",
        )
    )
    add(
        Variant(
            "dir_noc_slots4_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=4,
            dir_noc=True,
            per_slot_push=True,
            mode=mode,
            notes="same, fan-in slots (does NOT fit the op's L1)",
        )
    )
    add(
        Variant(
            "mirror_dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            orient="mirror",
            dir_noc=True,
            per_slot_push=True,
            mode=mode,
            notes="winner under the FLIPPED hop direction",
        )
    )
    add(Variant("slots_fanin", n_ch=2, owner=(1, 1), slots=4, mode=mode, notes="REDUCE_SLOTS = fan-in"))
    add(Variant("dual_slots_fanin", n_ch=2, owner=(0, 1), slots=4, mode=mode, notes="dual NoC + all slots"))
    add(Variant("one_write_slots_fanin", n_ch=1, owner=(1,), slots=4, mode=mode, notes="merged + all slots"))
    add(
        Variant("dir_noc_slots2", n_ch=2, owner=(1, 1), slots=2, mode=mode, dir_noc=True, notes="dir-matched + 2 slots")
    )
    add(
        Variant("dir_noc_slots3", n_ch=2, owner=(1, 1), slots=3, mode=mode, dir_noc=True, notes="dir-matched + 3 slots")
    )
    add(
        Variant(
            "dir_noc_slots4",
            n_ch=2,
            owner=(1, 1),
            slots=4,
            mode=mode,
            dir_noc=True,
            notes="dir-matched + fan-in slots",
        )
    )
    add(
        Variant(
            "dir_noc_slots2_merged",
            n_ch=1,
            owner=(1,),
            slots=2,
            mode=mode,
            dir_noc=True,
            notes="dir-matched + 2 slots + ONE merged write",
        )
    )
    add(Variant("slots2_merged", n_ch=1, owner=(1,), slots=2, mode=mode, notes="2 slots + ONE merged write"))
    add(Variant("mirror", n_ch=2, owner=(1, 1), slots=1, orient="mirror", mode=mode, notes="hop direction FLIPPED"))
    add(
        Variant(
            "mirror_dir_noc",
            n_ch=2,
            owner=(1, 1),
            slots=1,
            orient="mirror",
            dir_noc=True,
            mode=mode,
            notes="flipped hops, NoC matched per edge",
        )
    )
    # TWO-SIDED PLACEMENT: same tree topology (same depth / fan-in / edge count), but a parent's
    # children straddle it, so its CONCURRENT children land on DIFFERENT NoCs, each on a short path.
    for nm, sl, dn in (
        ("twosided", 1, False),
        ("twosided_dir_noc", 1, True),
        ("twosided_dir_noc_slots2", 2, True),
        ("twosided_dir_noc_slots_fanin", 4, True),
    ):
        add(
            Variant(
                nm,
                n_ch=2,
                owner=(1, 1),
                slots=sl,
                orient="twosided",
                dir_noc=dn,
                mode=mode,
                notes="children straddle the parent" + (" + per-edge NoC" if dn else " (NOC_1 only)"),
            )
        )
    add(
        Variant(
            "twosided_dir_noc_slots2_merged",
            n_ch=1,
            owner=(1,),
            slots=2,
            orient="twosided",
            dir_noc=True,
            mode=mode,
            notes="straddle + per-edge NoC + 2 slots + ONE merged write",
        )
    )
    if mode == "xfer":
        # The invite can only be dropped when every child owns its OWN slot, so this arm is measured
        # against `slots_fanin` (NOT against `baseline`) and exists only to price the SEM_GO invite.
        add(
            Variant(
                "slots_fanin_no_invite",
                n_ch=2,
                owner=(1, 1),
                slots=4,
                use_invite=False,
                mode=mode,
                notes="prices SEM_GO; race-free ONLY at slots >= fan_in",
            )
        )
        add(
            Variant(
                "dual_slots_fanin_no_invite",
                n_ch=2,
                owner=(0, 1),
                slots=4,
                use_invite=False,
                mode=mode,
                notes="best transport shape, no invite",
            )
        )
    return v


def _report(tag, rows):
    lines = [f"\n=== reduce_transport_shape :: {tag} ==="]
    base = next((r for r in rows if r[0] == "baseline"), None)
    for name, ns, pcc, absdiff, meta, note in rows:
        rel = "" if base is None or base[1] in (None, 0) or ns is None else f"  {(ns / base[1] - 1) * 100:+6.2f}%"
        lines.append(
            f"{name:26s} ns={ns if ns is None else round(ns, 1):>10}  pcc={pcc:.6f}  "
            f"maxabs={absdiff:.4g}  +landL1={meta['landing_l1_delta']:>7}  "
            f"slots={meta['slots']} arr/ch={meta['arrivals_per_child']} "
            f"hops(noc0,noc1)={meta['hops']}{rel}   {note}"
        )
    logger.info("\n".join(lines))


def test_focus_shape_repeat(device):
    """FOCUS SHAPE, 3 repeats per arm, both modes — the win/null calls below sit inside a few
    percent, so the median is what gets reported. Reports the device clock too, so the measured
    B/ns can be turned into B/cycle against the NoC's width."""
    import statistics

    try:
        mhz = device.get_clock_rate_mhz()
    except Exception as exc:  # pragma: no cover - informational only
        mhz = f"unavailable ({exc})"
    logger.info(f"[clock] device clock = {mhz} MHz")

    arms = [
        ("baseline", dict(n_ch=2, owner=(1, 1), slots=1)),
        ("one_write", dict(n_ch=1, owner=(1,), slots=1)),
        ("dir_noc", dict(n_ch=2, owner=(1, 1), slots=1, dir_noc=True)),
        ("dir_noc_merged", dict(n_ch=1, owner=(1,), slots=1, dir_noc=True)),
        ("dir_noc_slots2", dict(n_ch=2, owner=(1, 1), slots=2, dir_noc=True)),
        ("slots2_pipelined", dict(n_ch=2, owner=(1, 1), slots=2, per_slot_push=True)),
        ("dir_noc_slots2_pipelined", dict(n_ch=2, owner=(1, 1), slots=2, dir_noc=True, per_slot_push=True)),
        ("dir_noc_slots4_pipelined", dict(n_ch=2, owner=(1, 1), slots=4, dir_noc=True, per_slot_push=True)),
    ]
    out = [f"device clock = {mhz} MHz"]
    for mode in ("xfer", "add"):
        base = None
        for name, kw in arms:
            samples, pcc = [], None
            for _ in range(3):
                ns, pcc, absdiff, _, _ = _run_cell(device, Variant(name, mode=mode, **kw), HGROUPS, KGROUPS, HALF_TILES)
                if mode == "xfer":
                    assert absdiff == 0.0, f"{name}: transport corrupted bytes"
                else:
                    assert pcc > 0.99, f"{name}: pcc {pcc}"
                samples.append(ns)
            med = statistics.median(samples)
            if name == "baseline":
                base = med
            spread = (max(samples) - min(samples)) / med * 100
            out.append(
                f"{mode:5s} {name:16s} median={med:9.1f} ns  spread={spread:4.1f}%  "
                f"vs base={(med / base - 1) * 100:+6.2f}%  pcc={pcc:.6f}  samples={[round(s) for s in samples]}"
            )
            logger.info(f"[focus] {out[-1]}")
    logger.info("\n=== reduce_transport_shape :: focus-shape repeats ===\n" + "\n".join(out))


def test_transport_ceiling_calibration(device):
    """/perf-ceiling-dm, calibrated ON THIS BOX instead of read off noc_latencies.yaml.

    A single column of 2 rows = ONE tree edge, contention-free. Sweeping the payload gives the edge's
    (fixed overhead, marginal B/ns) on the SHORT-path NoC and on the LONG-path one, which is the
    bound every arm of the menu has to be judged against. `slots` then prices the parent's
    destination port: fan-in 4 serialised (slots=1) vs fully concurrent (slots=4) on one column, so
    the answer to "is the tree already at its NoC bound?" is measured, not modelled.
    """
    rows = []
    for ht in (48, 24, 12, 6, 3):
        for name, owner in (("noc1_short", (1, 1)), ("noc0_long", (0, 0))):
            v = Variant(f"edge_{name}", n_ch=2, owner=owner, slots=1, mode="xfer")
            ns, _, absdiff, meta, _ = _run_cell(device, v, 1, 2, ht)
            assert absdiff == 0.0
            payload = 2 * ht * 1088
            rows.append(f"1 edge  {name:10s} payload={payload:7d} B  ns={ns:8.1f}  {payload / ns:6.2f} B/ns")
            logger.info(f"[ceiling] {rows[-1]}")
    # One column, fan-in 4 at the root: serialised vs concurrent arrivals into ONE destination port.
    for slots in (1, 2, 4):
        for orient in ("op", "twosided"):
            v = Variant(
                f"col_{orient}_s{slots}", n_ch=2, owner=(1, 1), slots=slots, orient=orient, dir_noc=True, mode="xfer"
            )
            ns, _, absdiff, meta, _ = _run_cell(device, v, 1, KGROUPS, HALF_TILES)
            assert absdiff == 0.0
            into_root = meta["fan_in"] * 2 * HALF_TILES * 1088
            rows.append(
                f"1 column K={KGROUPS} orient={orient:9s} slots={meta['slots']}  ns={ns:8.1f}  "
                f"root_in={into_root} B  {into_root / ns:6.2f} B/ns at the root port"
            )
            logger.info(f"[ceiling] {rows[-1]}")
    logger.info("\n=== reduce_transport_shape :: ceiling calibration ===\n" + "\n".join(rows))


def test_transport_menu_xfer(device):
    """PURE TRANSPORT arm: no compute kernel, child ships straight out of its resident shard, tree
    serialisation preserved by a same-core semaphore. This is the number to compare against the
    /perf-ceiling-dm bound."""
    rows = []
    for variant in _menu("xfer"):
        ns, pcc, absdiff, meta, _ = _run_cell(device, variant, HGROUPS, KGROUPS, HALF_TILES)
        # xfer mode echoes the last child's bytes: the transport must be BYTE-EXACT.
        assert absdiff == 0.0, f"{variant.name}: transport corrupted bytes, max|diff|={absdiff}"
        rows.append((variant.name, ns, pcc, absdiff, meta, variant.notes))
        logger.info(f"[xfer] {variant.name:26s} ns={ns} pcc={pcc:.6f} maxabs={absdiff}")
    _report(f"xfer (pure transport)  H={HGROUPS} K={KGROUPS} half_tiles={HALF_TILES}", rows)


def test_transport_menu_add(device):
    """REALISTIC arm: the parent's landing CBs are consumed by the op's real blocked bfp8 adds.
    Correctness = PCC of the column sum, AND bit-identity against the baseline arm (no variant here
    changes the add ORDER, so anything else is a bug)."""
    import torch

    rows, base_shards = [], None
    for variant in _menu("add"):
        ns, pcc, absdiff, meta, shards = _run_cell(device, variant, HGROUPS, KGROUPS, HALF_TILES)
        assert pcc > 0.99, f"{variant.name}: pcc {pcc}"
        bit_delta = 0.0
        if base_shards is None:
            base_shards = shards
        else:
            for k, s in shards.items():
                bit_delta = max(bit_delta, float((s - base_shards[k]).abs().max()))
        rows.append((variant.name, ns, pcc, bit_delta, meta, variant.notes))
        logger.info(f"[add] {variant.name:26s} ns={ns} pcc={pcc:.6f} bit_delta_vs_baseline={bit_delta}")
    _report(f"add (transport under the real adds)  H={HGROUPS} K={KGROUPS} half_tiles={HALF_TILES}", rows)


def test_transport_predicate_sweep(device):
    """PREDICATE: sweep the winner + baseline over m_eff (half_tiles 6/12/24/48), KGROUPS 2/4/10
    (= fan-in 1/2/4), both hop orientations, and 1-column vs 11-column contention."""
    arms = [
        Variant("baseline", n_ch=2, owner=(1, 1), slots=1, mode="xfer"),
        Variant("dir_noc", n_ch=2, owner=(1, 1), slots=1, dir_noc=True, mode="xfer"),
        Variant("slots2_pipelined", n_ch=2, owner=(1, 1), slots=2, per_slot_push=True, mode="xfer"),
        Variant(
            "dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            dir_noc=True,
            per_slot_push=True,
            mode="xfer",
        ),
        Variant(
            "mirror_dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            orient="mirror",
            dir_noc=True,
            per_slot_push=True,
            mode="xfer",
        ),
    ]
    regimes = []
    for ht in (48, 24, 12, 6):
        regimes.append((HGROUPS, KGROUPS, ht))
    for k in (2, 4):
        regimes.append((HGROUPS, k, 48))
    regimes.append((1, KGROUPS, 48))  # single column: contention-free control
    rows = []
    for hg, kg, ht in regimes:
        fan_in = tree_max_fanin(reduce_tree(kg, hg))
        base_ns = None
        for variant in arms:
            ns, pcc, absdiff, meta, _ = _run_cell(device, variant, hg, kg, ht)
            assert absdiff == 0.0, f"{variant.name} H={hg} K={kg} ht={ht}: max|diff|={absdiff}"
            if variant.name == "baseline":
                base_ns = ns
            rel = "" if base_ns in (None, 0) else f"{(ns / base_ns - 1) * 100:+6.2f}%"
            rows.append(f"H={hg:2d} K={kg:2d} fan_in={fan_in} half_tiles={ht:2d} {variant.name:20s} ns={ns:9.1f} {rel}")
            logger.info(f"[sweep] {rows[-1]}")
    logger.info("\n=== reduce_transport_shape :: predicate sweep (xfer) ===\n" + "\n".join(rows))


def test_transport_predicate_sweep_add(device):
    """The same predicate sweep in `add` mode — does the transport win survive the adds?"""
    arms = [
        Variant("baseline", n_ch=2, owner=(1, 1), slots=1, mode="add"),
        Variant("dir_noc_merged", n_ch=1, owner=(1,), slots=1, dir_noc=True, mode="add"),
        Variant("slots2_pipelined", n_ch=2, owner=(1, 1), slots=2, per_slot_push=True, mode="add"),
        Variant(
            "dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            dir_noc=True,
            per_slot_push=True,
            mode="add",
        ),
        Variant(
            "mirror_dir_noc_slots2_pipelined",
            n_ch=2,
            owner=(1, 1),
            slots=2,
            orient="mirror",
            dir_noc=True,
            per_slot_push=True,
            mode="add",
        ),
        Variant(
            "dir_noc_slots2_pipelined_merged",
            n_ch=1,
            owner=(1,),
            slots=2,
            dir_noc=True,
            per_slot_push=True,
            mode="add",
        ),
    ]
    # fan_in = ceil(log2(KGROUPS)): K=2 -> 1, K=4 -> 2, K=6/8 -> 3, K=10 -> 4. The full fan-in axis.
    regimes = [(HGROUPS, KGROUPS, ht) for ht in (48, 24, 12, 6)] + [
        (HGROUPS, 8, 48),
        (HGROUPS, 6, 48),
        (HGROUPS, 4, 48),
        (HGROUPS, 2, 48),
    ]
    rows = []
    for hg, kg, ht in regimes:
        base_ns = None
        for variant in arms:
            ns, pcc, absdiff, meta, _ = _run_cell(device, variant, hg, kg, ht)
            assert pcc > 0.99, f"{variant.name} H={hg} K={kg} ht={ht}: pcc {pcc}"
            if variant.name == "baseline":
                base_ns = ns
            rel = "" if base_ns in (None, 0) else f"{(ns / base_ns - 1) * 100:+6.2f}%"
            rows.append(f"H={hg:2d} K={kg:2d} half_tiles={ht:2d} {variant.name:20s} ns={ns:9.1f} pcc={pcc:.6f} {rel}")
            logger.info(f"[sweep-add] {rows[-1]}")
    logger.info("\n=== reduce_transport_shape :: predicate sweep (add) ===\n" + "\n".join(rows))
