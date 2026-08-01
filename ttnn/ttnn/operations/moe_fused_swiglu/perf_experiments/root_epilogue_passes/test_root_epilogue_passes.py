# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: how few blocked DEST passes can moe_fused_swiglu's reduce-ROOT epilogue run in?

    h = SiLU(gate_acc + last_gate_child) * (up_acc + last_up_child)

Single core, compute-only (sharded L1, no NoC transport). Correctness is the only pass/fail; perf is
measured (DEVICE KERNEL DURATION [ns] via ReadDeviceProfiler) and reported, never asserted. See
bench.py's module docstring for the arm menu and the precision contract.

    scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/root_epilogue_passes/test_root_epilogue_passes.py

Env knobs: REF_TRIALS (3), REF_KERNEL_ITERS (20), REF_ARMS (csv), REF_REGIMES (csv of m_eff:hn_pad),
REF_REPORT (path to write the markdown table).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

# NOTE: `torch` is imported LAZILY (scripts/validate_no_global_torch_imports.py forbids a
# module-level torch import anywhere under ttnn/ttnn/).
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.root_epilogue_passes.bench import (
    ARMS,
    BASELINE,
    SCRATCH_TILES_OP,
    VARIANTS,
    create_sharded_memory_config,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
PCC_THRESHOLD = 0.975  # feature_spec.py pcc_threshold (soft gate)
BFP8_TILE_BYTES = 1088

# The focus shape's derived epilogue geometry: m_eff=8 token tile-rows x HN_PAD=6 hidden tiles.
FOCUS = (8, 6)
# Predicate sweep: m_eff 1/2/4/8 (6/12/24/48 tiles) x HN_PAD 6, plus the ragged last column HN_PAD=4.
REGIMES = [(1, 6), (2, 6), (4, 6), (8, 6), (4, 4), (8, 4)]
# Arms measured at every regime (the whole menu runs at the focus shape only).
SWEEP_ARMS = (
    "baseline",
    "hoist_rows",
    "blk_packer",
    "add_silu_chain",
    "add_silu_chain_nr",
    "fuse_silu_mul",
    "sigappx_mul",
    "sigappx_fused",
    "single_pass",
)


# =============================================================================
# Reference construction — bfp8-dequantized inputs, so the reference isolates KERNEL error from
# the bfloat8_b format's own quantization (the op's own precision-baseline methodology).
# =============================================================================
def _make_case(device, m_eff, hn_pad, seed=7, scale=None):
    import torch

    torch.manual_seed(seed)
    m, n = m_eff * TILE, hn_pad * TILE
    mem_cfg = create_sharded_memory_config((m, n))
    # Operand scale. The SFPLUT approx sigmoid's error is input-RANGE dependent (it is a 3-coefficient
    # LUT), so the *_sigappx_* arms' PCC has to be quoted per scale, not once. REF_SCALE sweeps it.
    if scale is None:
        scale = float(os.environ.get("REF_SCALE", "0.5"))

    def _dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg)

    tensors = [_dev(torch.randn(m, n) * scale) for _ in range(4)]
    g0, u0, g1, u1 = (ttnn.to_torch(t).to(torch.float32) for t in tensors)
    refs = {
        "epilogue": torch.nn.functional.silu(g0 + g1) * (u0 + u1),
        "gate_silu": torch.nn.functional.silu(g0 + g1),
        "add": g0 + g1,
    }
    return tensors, refs


def _pcc(actual, expected):
    import torch

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    if torch.all(a == a[0]) or torch.all(e == e[0]):
        return 1.0 if torch.allclose(a, e) else 0.0
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


# Per-reference-kind PCC floor. `epilogue` is the op's real output and carries the op's own soft
# gate. `gate_silu` is a stage-(a)-only artefact: SiLU(gate_sum) spans zero with a long
# small-magnitude tail, and storing THAT in bfloat8_b (shared exponent per 16-value block) costs
# ~0.96 by itself — a property of the fixture, not of any arm. The sharp gate for those arms is the
# bit-comparison against the shipped stage-(a) implementation, reported alongside.
_MIN_PCC = {"epilogue": PCC_THRESHOLD, "gate_silu": 0.95, "add": PCC_THRESHOLD}

# Arms that deliberately change HOW the same function is evaluated on the SFPU/FPU and are expected
# to move PCC. They are MENU ENTRIES priced with their precision cost, never silent substitutions,
# so their gate is only a "did the kernel compute the right function at all" sanity floor. Whether
# they clear the op's 0.975 soft gate is REPORTED, not asserted.
PRECISION_PRICED = {"sigacc_mul", "sigappx_mul", "sigappx_fused", "single_pass_sigappx", "a_only_sigappx"}
_SANITY_PCC = 0.90


def _check(arm, output, refs, label):
    """Returns (pcc, torch tensor of the device output); pcc is None for ref-less arms."""
    import torch

    kind = ARMS[arm]["ref"]
    actual = ttnn.to_torch(output).to(torch.float32)
    if kind is None:
        return None, actual
    pcc = _pcc(actual, refs[kind])
    floor = _SANITY_PCC if arm in PRECISION_PRICED else _MIN_PCC[kind]
    assert pcc >= floor, f"{arm} {label}: PCC {pcc:.6f} < {floor}"
    if arm in PRECISION_PRICED and pcc < _MIN_PCC[kind]:
        logger.warning(
            f"{arm} {label}: PCC {pcc:.6f} is BELOW the op's soft gate {_MIN_PCC[kind]} — priced, not chosen"
        )
    return pcc, actual


# =============================================================================
# In-process device-kernel timing (validated pattern from examples/compute_block_size).
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard the warm-up window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard the first timed pass
                samples[name].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _int(name, default):
    return int(os.environ.get(name, default))


def _blk(arm):
    """The op's graduated ELTWISE_BLK (8), capped per arm. See bench.py's DEST-reuse window rule."""
    return min(_int("REF_BLK", "8"), ARMS[arm]["blk_cap"])


def _arms(default):
    if raw := os.environ.get("REF_ARMS"):
        return tuple(a.strip() for a in raw.split(",") if a.strip())
    return default


def _regimes():
    if raw := os.environ.get("REF_REGIMES"):
        return [tuple(int(v) for v in cell.split(":")) for cell in raw.split(",") if cell.strip()]
    return REGIMES


def _l1_scratch_bytes(arm):
    """Scratch L1 the arm needs at the op's own sizing (48 bfp8 tiles per intermediate CB).

    The op today allocates cb_gate_silu (48 tiles = 52,224 B) and does the up add in place, so
    `baseline`'s op-side scratch is gate_silu alone. An arm that never materializes gate_silu
    returns 52,224 B to the L1 budget.
    """
    scratch = set(ARMS[arm]["scratch"])
    # The op's up sum is IN-PLACE on cb_up_acc, so `up_sum` costs nothing in the op; `gate_sum` is a
    # genuinely new CB an arm would have to add.
    billed = {"gate_silu", "gate_sum"} & scratch
    return len(billed) * SCRATCH_TILES_OP * BFP8_TILE_BYTES


# =============================================================================
# Correctness — every arm, focus shape; then the swept arms at every regime.
# =============================================================================
def test_root_epilogue_passes_correctness(device):
    import torch

    regs = _regimes()
    for m_eff, hn_pad in ([FOCUS] if FOCUS in regs else []) + [r for r in regs if r != FOCUS]:
        arms = _arms(VARIANTS if (m_eff, hn_pad) == FOCUS else SWEEP_ARMS)
        tensors, refs = _make_case(device, m_eff, hn_pad)
        # Per-ref-kind SHIPPED-implementation output, for the bit-comparison (first arm of each kind).
        anchors = {}
        for arm in arms:
            out = run_op(tensors, m_eff=m_eff, hn_pad=hn_pad, arm=arm, kernel_iters=1, blk=_blk(arm))
            pcc, actual = _check(arm, out, refs, f"m_eff={m_eff} hn_pad={hn_pad}")
            kind = ARMS[arm]["ref"]
            tag = f"PCC={pcc:.6f}" if pcc is not None else "PCC=n/a"
            if kind is not None:
                if kind not in anchors:
                    anchors[kind] = (arm, actual)
                anchor_arm, anchor = anchors[kind]
                delta = (actual - anchor).abs().max().item()
                bit = f"bit-identical to {anchor_arm}" if delta == 0.0 else f"max|d| vs {anchor_arm}={delta:.5g}"
                tag = f"{tag}  {bit}"
            logger.info(f"m_eff={m_eff} hn_pad={hn_pad}  {arm:22s} {tag}")


# =============================================================================
# Diagnostic — WHICH tiles does a DEST-reuse chain get wrong, and at what DEST window width?
# Not a perf test; it exists to characterise the eltwise_chain DEST-reuse defect this bench hit, so
# the report can state the predicate instead of guessing at a mechanism. Select with
# `-k diag_dest_reuse`.
# =============================================================================
def test_diag_dest_reuse_pattern(device):
    import torch

    m_eff, hn_pad = FOCUS
    tensors, refs = _make_case(device, m_eff, hn_pad)
    expected = refs["epilogue"]
    blks = [int(v) for v in os.environ.get("REF_DIAG_BLKS", "1,2,4,6,8").split(",")]
    diag_arms = _arms(("baseline", "fuse_silu_mul", "sigacc_mul"))
    for blk in blks:
        for arm in diag_arms:
            out = run_op(tensors, m_eff=m_eff, hn_pad=hn_pad, arm=arm, kernel_iters=1, blk=blk)
            actual = ttnn.to_torch(out).to(torch.float32)
            pcc = _pcc(actual, expected)
            # per-tile max |diff|, tile t = row_tile * hn_pad + col_tile (the CB's linear order)
            per_tile = []
            for i in range(m_eff):
                for j in range(hn_pad):
                    a = actual[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
                    e = expected[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
                    per_tile.append((a - e).abs().max().item())
            bad = [t for t, d in enumerate(per_tile) if d > 0.05]
            logger.info(
                f"blk={blk} {arm:16s} PCC={pcc:.6f}  tiles with max|d|>0.05: {len(bad)}/{len(per_tile)}  "
                f"first={bad[:12]}"
            )


# =============================================================================
# Perf — the whole menu at the focus shape, the shortlist across the predicate sweep.
# =============================================================================
def test_root_epilogue_passes_device_perf(device):
    trials = _int("REF_TRIALS", "3")
    kernel_iters = _int("REF_KERNEL_ITERS", "20")
    regimes = _regimes()

    lines = [
        "# root_epilogue_passes — isolated bake-off (single core, compute-only)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters}",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] per ONE root-epilogue evaluation "
        "(h = SiLU(gate_acc+last_gate_child) * (up_acc+last_up_child)).",
        "`overhead` = per-iteration CB scaffolding only; `plain_add_x8` = the root's 8 plain blocked",
        "48-tile reduce adds; `a_only_*` = stage (a) alone. L1 = op-side scratch the arm needs.",
        "",
    ]

    for m_eff, hn_pad in regimes:
        arms = _arms(VARIANTS if (m_eff, hn_pad) == FOCUS else SWEEP_ARMS)
        tensors, refs = _make_case(device, m_eff, hn_pad)

        pccs, deltas, anchors = {}, {}, {}
        for arm in arms:
            out = run_op(tensors, m_eff=m_eff, hn_pad=hn_pad, arm=arm, kernel_iters=1, blk=_blk(arm))
            pcc, actual = _check(arm, out, refs, f"m_eff={m_eff} hn_pad={hn_pad}")
            pccs[arm] = pcc
            kind = ARMS[arm]["ref"]
            if kind is not None:
                anchors.setdefault(kind, (arm, actual))
                deltas[arm] = (actual - anchors[kind][1]).abs().max().item()

        runners = {
            arm: (
                lambda m_eff=m_eff, hn_pad=hn_pad, arm=arm: run_op(
                    tensors, m_eff=m_eff, hn_pad=hn_pad, arm=arm, kernel_iters=kernel_iters, blk=_blk(arm)
                )
            )
            for arm in arms
        }
        samples = _measure(device, runners, trials, kernel_iters)

        base_med = statistics.median(samples[BASELINE]) if BASELINE in samples else None
        oh = statistics.median(samples["overhead"]) if "overhead" in samples else None
        focus_tag = "  <-- FOCUS" if (m_eff, hn_pad) == FOCUS else ""
        lines.append(f"## m_eff={m_eff}, HN_PAD={hn_pad}, block_tiles={m_eff * hn_pad}{focus_tag}")
        lines.append("")
        lines.append(
            "| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in arms:
            med = statistics.median(samples[arm])
            net = f"{med - oh:.0f}" if oh is not None else "-"
            speed = f"{base_med / med:.3f}x" if base_med else "-"
            pcc = f"{pccs[arm]:.6f}" if pccs.get(arm) is not None else "-"
            delta = f"{deltas[arm]:.4g}" if arm in deltas else "-"
            lines.append(f"| {arm} | {med:.0f} | {net} | {speed} | {pcc} | {delta} | {_l1_scratch_bytes(arm):,} |")
        lines.append("")
        logger.info(f"m_eff={m_eff} hn_pad={hn_pad}: baseline={base_med:.0f} ns  overhead={oh}")
        for arm in arms:
            logger.info(f"    {arm:22s} {statistics.median(samples[arm]):>10.0f} ns   {ARMS[arm]['note']}")

    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if path := os.environ.get("REF_REPORT"):
        Path(path).write_text(report)
