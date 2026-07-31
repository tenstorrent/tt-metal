# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for moe_fused_swiglu's reduce-accumulate mechanism.

See bench.py's module docstring for the full writeup of the three mechanisms (baseline /
pack_l1_acc / dest_acc) and the raw-LLK justification.

Correctness: device output (read back as float32) vs a full-fp32 torch reference
(seed + sum(children), no intermediate quantization) via PCC -- the SAME reference for every
variant, so the PCC comparison directly shows each mechanism's OWN quantization cost.

Perf: `scripts/run_safe_pytest.sh --run-all
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/reduce_accum_mechanism/test_reduce_accum_mechanism.py`
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
from pathlib import Path

import pytest
from loguru import logger

# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn

from models.common.utility_functions import comp_pcc

# Loaded by explicit file path (not a package-dotted import) so this experiment stays fully
# self-contained under perf_experiments/reduce_accum_mechanism/ -- no __init__.py needed in the
# shared perf_experiments/ parent, which would risk colliding with sibling part-optimizers' own
# idea dirs (see gateup_in0_share/test_gateup_in0_share.py for the same pattern).
_MOD_PATH = Path(__file__).resolve().parent / "bench.py"
_spec = importlib.util.spec_from_file_location("reduce_accum_mechanism_bench", _MOD_PATH)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

VARIANT_BASELINE = _bench.VARIANT_BASELINE
VARIANT_PACK_L1_ACC = _bench.VARIANT_PACK_L1_ACC
VARIANT_DEST_ACC = _bench.VARIANT_DEST_ACC
VARIANT_NAMES = _bench.VARIANT_NAMES
make_seed_and_children = _bench.make_seed_and_children
run_reduce_accum = _bench.run_reduce_accum
create_fuse_program_descriptor = _bench.create_fuse_program_descriptor
create_sharded_memory_config = _bench.create_sharded_memory_config
default_compute_kernel_config = _bench.default_compute_kernel_config

TILE = 32
PCC_THRESHOLD = 0.975  # soft gate carried in the assignment (the op's own feature_spec floor)
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Predicate sweep: fan-in (root=4, r=4 has 2, r=2/6/8 have 1) x block tile count
# (48 = focus shape m_eff*HN_PAD; 24 = m_eff 4/count 128; 6 = m_eff 1/count 32).
FAN_INS = (1, 2, 4)
BLOCK_TILES_LIST = (48, 24, 6)
FOCUS_FAN_IN = 4
FOCUS_BLOCK_TILES = 48


def _expected(seed, children):
    import torch

    ref = seed.to(torch.float32).clone()
    for c in children:
        ref = ref + c.to(torch.float32)
    return ref


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


def _measure_once(device, run_fn):
    """One fresh-cache run (perf-measure discipline: device kernel time has no warm-up
    transient); the caller may re-invoke for a median only on a borderline/implausible result."""
    out = run_fn()
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)
    return out, ns


# pack_l1_acc is only VALID at a linear accumulator format (bf16/fp16/fp32): the packer's raw
# L1-accumulate hardware register does a straightforward add, which is meaningless on bfp8_b
# (shared-exponent block floating point) -- confirmed on device (probes/probe_048.py): identical
# random data, bf16 max_abs_diff=0.0065 (quantization noise, correct), bfp8_b max_abs_diff=1.17
# (badly wrong). This exactly matches the real op's own design note (op_design.md 4.1:
# "packer_l1_acc forces >= fp16_b for partials") -- the real op only ever uses packer_l1_acc on
# its bf16 cb_*_interm CBs, never on the bfp8_b cb_gate_acc/cb_reduce_*_in CBs. So pack_l1_acc is
# NOT an independent option: it can only be graduated TOGETHER WITH the bf16 intermediate-format
# change (option 4), never against the op's current bfp8_b accumulator.
def _dtype_for(variant):
    return ttnn.bfloat16 if variant == VARIANT_PACK_L1_ACC else ttnn.bfloat8_b


@pytest.mark.parametrize("variant", [VARIANT_BASELINE, VARIANT_PACK_L1_ACC, VARIANT_DEST_ACC])
@pytest.mark.parametrize("fan_in", FAN_INS)
@pytest.mark.parametrize("block_tiles", BLOCK_TILES_LIST)
def test_reduce_accum_correctness(device, variant, fan_in, block_tiles):
    """All three mechanisms compute the same seed+sum(children); PCC differs only by how many
    times the running value is requantized along the way (and, for pack_l1_acc, at its only VALID
    format -- see _dtype_for)."""
    import torch

    dtype = _dtype_for(variant)
    tt_seed, tt_children, seed, children = make_seed_and_children(device, fan_in, block_tiles, dtype)
    expected = _expected(seed, children)
    out = ttnn.to_torch(
        run_reduce_accum(tt_seed, tt_children, variant=variant, fan_in=fan_in, block_tiles=block_tiles, dtype=dtype)
    ).to(torch.float32)
    ok, pcc_val = comp_pcc(expected, out, PCC_THRESHOLD)
    assert (
        ok
    ), f"{VARIANT_NAMES[variant]} fan_in={fan_in} block_tiles={block_tiles} dtype={dtype}: pcc={pcc_val} < {PCC_THRESHOLD}"


def test_reduce_accum_device_perf_sweep(device):
    """The core deliverable: baseline vs pack_l1_acc vs dest_acc, device ns + PCC, across the
    fan-in x block-tiles predicate sweep. One fresh-cache run per (variant, fan_in, block_tiles).
    baseline/dest_acc measured at the op's current bfp8_b accumulator; pack_l1_acc measured at its
    only valid format, bf16 (see _dtype_for) -- reported as a DIFFERENT L1 cost/format, not silently
    normalized away."""
    import torch

    results = {}
    for block_tiles in BLOCK_TILES_LIST:
        for fan_in in FAN_INS:
            tt_seed8, tt_children8, seed8, children8 = make_seed_and_children(
                device, fan_in, block_tiles, ttnn.bfloat8_b
            )
            tt_seed16, tt_children16, seed16, children16 = make_seed_and_children(
                device, fan_in, block_tiles, ttnn.bfloat16
            )
            expected8 = _expected(seed8, children8)
            expected16 = _expected(seed16, children16)
            for variant in (VARIANT_BASELINE, VARIANT_PACK_L1_ACC, VARIANT_DEST_ACC):
                dtype = _dtype_for(variant)
                s, c, expected = (
                    (tt_seed16, tt_children16, expected16)
                    if dtype == ttnn.bfloat16
                    else (tt_seed8, tt_children8, expected8)
                )
                out, ns = _measure_once(
                    device,
                    lambda s=s, c=c, v=variant, f=fan_in, b=block_tiles, d=dtype: run_reduce_accum(
                        s, c, variant=v, fan_in=f, block_tiles=b, dtype=d
                    ),
                )
                assert ns is not None, f"profiler produced no data for {VARIANT_NAMES[variant]}"
                torch_out = ttnn.to_torch(out).to(torch.float32)
                _, pcc_val = comp_pcc(expected, torch_out, PCC_THRESHOLD)
                results[(block_tiles, fan_in, variant)] = (ns, pcc_val, dtype)

    lines = [
        "",
        "=== reduce_accum_mechanism sweep ===",
        f"{'block_tiles':>11} {'fan_in':>6} {'variant':>12} {'dtype':>9} {'ns':>10} {'pcc':>10}",
    ]
    for block_tiles in BLOCK_TILES_LIST:
        for fan_in in FAN_INS:
            for variant in (VARIANT_BASELINE, VARIANT_PACK_L1_ACC, VARIANT_DEST_ACC):
                ns, pcc_val, dtype = results[(block_tiles, fan_in, variant)]
                lines.append(
                    f"{block_tiles:>11} {fan_in:>6} {VARIANT_NAMES[variant]:>12} {str(dtype):>9} {ns:>10.1f} {pcc_val}"
                )
    logger.info("\n".join(lines))

    # Sanity: baseline should never be faster than pack_l1_acc/dest_acc at fan_in >= 2 (documented
    # mechanism direction) -- reported as data, not asserted as a pass/fail gate.
    base = results[(FOCUS_BLOCK_TILES, FOCUS_FAN_IN, VARIANT_BASELINE)][0]
    pl1 = results[(FOCUS_BLOCK_TILES, FOCUS_FAN_IN, VARIANT_PACK_L1_ACC)][0]
    dacc = results[(FOCUS_BLOCK_TILES, FOCUS_FAN_IN, VARIANT_DEST_ACC)][0]
    logger.info(
        f"focus shape (block_tiles={FOCUS_BLOCK_TILES}, fan_in={FOCUS_FAN_IN}): "
        f"baseline={base:.0f}ns pack_l1_acc(bf16)={pl1:.0f}ns ({base/pl1:.2f}x) dest_acc={dacc:.0f}ns ({base/dacc:.2f}x)"
    )


def test_reduce_accum_bf16_intermediate(device):
    """Option 4: hold the accumulator as bf16 in L1 instead of bfp8_b (an internal partial-CB
    format choice, not a change to the op's input/output/weight dtypes or precision knobs).
    Reports ns + PCC for baseline and pack_l1_acc at the focus shape."""
    import torch

    results = {}
    for dtype, name in ((ttnn.bfloat8_b, "bfp8_b"), (ttnn.bfloat16, "bf16")):
        tt_seed, tt_children, seed, children = make_seed_and_children(device, FOCUS_FAN_IN, FOCUS_BLOCK_TILES, dtype)
        expected = _expected(seed, children)
        for variant in (VARIANT_BASELINE, VARIANT_PACK_L1_ACC):
            out, ns = _measure_once(
                device,
                lambda s=tt_seed, c=tt_children, v=variant, d=dtype: run_reduce_accum(
                    s, c, variant=v, fan_in=FOCUS_FAN_IN, block_tiles=FOCUS_BLOCK_TILES, dtype=d
                ),
            )
            assert ns is not None
            torch_out = ttnn.to_torch(out).to(torch.float32)
            _, pcc_val = comp_pcc(expected, torch_out, PCC_THRESHOLD)
            results[(name, variant)] = (ns, pcc_val)

    lines = ["", "=== reduce_accum_mechanism: accumulator intermediate format ==="]
    for name in ("bfp8_b", "bf16"):
        for variant in (VARIANT_BASELINE, VARIANT_PACK_L1_ACC):
            ns, pcc_val = results[(name, variant)]
            lines.append(f"  acc_format={name:<8} variant={VARIANT_NAMES[variant]:<12} ns={ns:.1f} pcc={pcc_val}")
    logger.info("\n".join(lines))


def test_reduce_accum_fuse_gate_up(device):
    """Option 5: fuse the gate+up pack_l1_acc call into ONE chain call (2x tiles) vs two separate
    role calls (gate then up, same launch) -- isolates the per-eltwise_chain-call overhead.
    Uses bf16 throughout: pack_l1_acc is only valid at a linear accumulator format (see
    _dtype_for / the module docstring bug note) -- bfp8_b would silently corrupt this measurement."""
    import torch

    fan_in, block_tiles = FOCUS_FAN_IN, FOCUS_BLOCK_TILES

    # Unfused: two independent role blocks in one launch.
    tt_seed_a, tt_children_a, seed_a, children_a = make_seed_and_children(
        device, fan_in, block_tiles, ttnn.bfloat16, seed_val=1
    )
    tt_seed_b, tt_children_b, seed_b, children_b = make_seed_and_children(
        device, fan_in, block_tiles, ttnn.bfloat16, seed_val=2
    )
    acc_a = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, block_tiles * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(block_tiles),
    )
    acc_b = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, block_tiles * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(block_tiles),
    )
    desc, tensors = create_fuse_program_descriptor(
        tt_seed_a, tt_children_a, acc_a, tt_seed_b, tt_children_b, acc_b, fan_in=fan_in, block_tiles=block_tiles
    )
    out = ttnn.generic_op(tensors, desc)
    ttnn.synchronize_device(device)
    unfused_ns = _read_kernel_ns(device)
    assert unfused_ns is not None
    expected_a = _expected(seed_a, children_a)
    expected_b = _expected(seed_b, children_b)
    torch_a = ttnn.to_torch(acc_a).to(torch.float32)
    torch_b = ttnn.to_torch(acc_b).to(torch.float32)
    _, pcc_a = comp_pcc(expected_a, torch_a, PCC_THRESHOLD)
    _, pcc_b = comp_pcc(expected_b, torch_b, PCC_THRESHOLD)

    # Free the unfused arm's tensors before allocating the fused arm's -- everything here is
    # single-core sharded L1 (~1.4 MB total), and the two arms' tensors together would overflow it.
    for t in (tt_seed_a, *tt_children_a, acc_a, tt_seed_b, *tt_children_b, acc_b):
        ttnn.deallocate(t)

    # Fused: ONE combined pack_l1_acc call over 2*block_tiles (gate then up concatenated).
    import torch as _torch

    seed_ab = _torch.cat([seed_a, seed_b], dim=1)
    children_ab = [_torch.cat([ca, cb], dim=1) for ca, cb in zip(children_a, children_b)]
    cfg = create_sharded_memory_config(2 * block_tiles)
    tt_seed_ab = ttnn.from_torch(
        seed_ab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg
    )
    tt_children_ab = [
        ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
        for c in children_ab
    ]
    out_fused, fused_ns = _measure_once(
        device,
        lambda: run_reduce_accum(
            tt_seed_ab,
            tt_children_ab,
            variant=VARIANT_PACK_L1_ACC,
            fan_in=fan_in,
            block_tiles=2 * block_tiles,
            dtype=ttnn.bfloat16,
        ),
    )
    assert fused_ns is not None
    expected_ab = _expected(seed_ab, children_ab)
    torch_ab = ttnn.to_torch(out_fused).to(torch.float32)
    _, pcc_ab = comp_pcc(expected_ab, torch_ab, PCC_THRESHOLD)

    logger.info(
        "\n=== reduce_accum_mechanism: fuse gate+up (pack_l1_acc, fan_in=4, block_tiles=48) ===\n"
        f"  unfused (2 role calls, 1 launch): {unfused_ns:.1f} ns   pcc_gate={pcc_a} pcc_up={pcc_b}\n"
        f"  fused   (1 combined call, 96 tiles): {fused_ns:.1f} ns   pcc={pcc_ab}\n"
        f"  delta: {(unfused_ns - fused_ns) / unfused_ns * 100:.2f}%"
    )
