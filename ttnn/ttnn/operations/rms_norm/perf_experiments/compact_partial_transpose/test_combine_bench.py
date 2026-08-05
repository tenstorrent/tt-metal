# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bake-off driver: correctness gate (the ONLY pass/fail) + measured sweep.

    correctness:  scripts/run_safe_pytest.sh --run-all <this file> -k correctness
    perf:         scripts/run_safe_pytest.sh --profile <this file> -k perf_sweep
                  then: python3 <this dir>/read_perf.py

The perf test issues ONE generic_op launch per (variant, GROUP_SIZE, rows, iters) in a fixed order
and writes that order to perf_manifest.csv; read_perf.py joins it against the profiler's
DEVICE KERNEL DURATION [ns] column by launch order and reports the per-iteration cost
(t_N - t_1) / (N - 1), which cancels the fixed kernel-launch floor.
"""

import csv
import pathlib

import pytest

import ttnn

from ttnn.operations.rms_norm.perf_experiments.compact_partial_transpose import cases, combine_bench

HERE = pathlib.Path(__file__).parent

ITERS_HI = 9
NONE, COPY = combine_bench.SEED_NONE, combine_bench.SEED_COPY

VARIANTS = ("base_phase0", "base_l1acc", "cand_root", "cand_recv", "member_pack", "member_copy", "recv_unpack")
MM_VARIANTS = ("cand_root", "member_pack", "recv_unpack")
ROWS_SWEEP = (1, 2, 4, 10, 32)
G_SWEEP = (4, 16, 32)
G_SWEEP_VARIANTS = ("base_l1acc", "cand_recv", "member_pack", "member_copy", "recv_unpack")


def _perf_configs():
    """(variant, group_size, rows, seed_mode, dest_batch) in launch order."""
    out = []
    for rows in ROWS_SWEEP:  # the rows sweep at the focus shape's GROUP_SIZE
        for v in VARIANTS:
            out.append((v, 8, rows, NONE, 4))
    for gs in G_SWEEP:  # the GROUP_SIZE sweep, decode (rows=1) and today's block (rows=10)
        for rows in (1, 10):
            for v in G_SWEEP_VARIANTS:
                out.append((v, gs, rows, NONE, 4))
    for rows in (10, 32):  # the portable-DEST-seed spelling of the matmul modes
        for v in MM_VARIANTS:
            out.append((v, 8, rows, COPY, 4))
    for rows in (10, 32):  # how much the un-pack's DEST batching is worth
        for b in (1, 2, 8):
            out.append(("recv_unpack", 8, rows, NONE, b))
            out.append(("cand_root", 8, rows, NONE, b))
    return out


def _correctness_configs():
    out = []
    for rows in (1, 10, 32):
        for v in VARIANTS:
            out.append((v, 8, rows, NONE, 4))
    for v in MM_VARIANTS:
        for rows in (10, 32):
            out.append((v, 8, rows, COPY, 4))
    for v in G_SWEEP_VARIANTS:
        out.append((v, 4, 2, NONE, 4))
        out.append((v, 32, 1, NONE, 4))
    for b in (1, 2, 8):  # the un-pack's DEST batching must not change the answer
        out.append(("recv_unpack", 8, 32, NONE, b))
        out.append(("cand_root", 8, 10, NONE, b))
    return out


def _alloc(device, variant, group_size, rows):
    import torch

    part_t, bank_t, expect = cases.make_case(variant, group_size, rows)
    in_pages, bank_pages, out_pages = combine_bench.geometry(variant, group_size, rows)
    assert part_t.shape[0] == in_pages * cases.TILE, (variant, part_t.shape, in_pages)
    assert bank_t.shape[0] == bank_pages * cases.TILE

    def dev(t, pages):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=combine_bench.single_core_shard(pages),
        )

    part = dev(part_t, in_pages)
    bank = dev(bank_t, bank_pages)
    out = dev(torch.full((out_pages * cases.TILE, cases.TILE), -7.0, dtype=torch.float32), out_pages)
    return part, bank, out, expect


def _launch(part, bank, out, variant, group_size, rows, seed_mode, iters, dest_batch=4):
    return combine_bench.run(
        part,
        bank,
        out,
        variant=variant,
        group_size=group_size,
        rows=rows,
        inv_w_bits=cases.INV_W_BITS,
        eps_bits=cases.EPS_BITS,
        iters=iters,
        seed=seed_mode,
        dest_batch=dest_batch,
    )


def _score(res, expect):
    """-> (worst op-equivalent pcc, worst rel-RMS, worst exact-zero violation)."""
    got = ttnn.to_torch(res).reshape(-1, cases.TILE)
    wp, wr, wz = 1.0, 0.0, 0.0
    for tile_idx, col, ref in expect:
        v = got[tile_idx * cases.TILE : (tile_idx + 1) * cases.TILE, col]
        if float(ref.abs().max()) == 0.0:
            wz = max(wz, float(v.abs().max()))
            continue
        wp = min(wp, cases.eff_pcc(v, ref))
        wr = max(wr, cases.rel_rms(v, ref))
    return wp, wr, wz


@pytest.mark.parametrize(
    "variant, group_size, rows, seed_mode, dest_batch",
    _correctness_configs(),
    ids=[f"{v}_g{g}_r{r}_s{s}_b{b}" for v, g, r, s, b in _correctness_configs()],
)
def test_correctness(device, variant, group_size, rows, seed_mode, dest_batch):
    """The only pass/fail in this experiment.  Perf is measured, never asserted."""
    part, bank, out, expect = _alloc(device, variant, group_size, rows)
    res = _launch(part, bank, out, variant, group_size, rows, seed_mode, 2, dest_batch)
    wp, wr, wz = _score(res, expect)
    print(f"\n{variant} G={group_size} rows={rows} seed={seed_mode}: pcc={wp:.7f} rel_rms={wr:.6f} zero_leak={wz:g}")
    assert wz == 0.0, f"positions that must be exactly 0 hold {wz}"
    assert wp > 0.9995, f"pcc {wp}"
    assert wr <= 0.04, f"rel_rms {wr}"


def test_perf_sweep(device):
    """One launch per (config, iters) in a deterministic order; the manifest records that order."""
    log = []
    for variant, group_size, rows, seed_mode, dest_batch in _perf_configs():
        part, bank, out, expect = _alloc(device, variant, group_size, rows)
        for iters in (1, ITERS_HI):
            _launch(part, bank, out, variant, group_size, rows, seed_mode, iters, dest_batch)
            ttnn.synchronize_device(device)
            log.append(
                dict(
                    idx=len(log),
                    variant=variant,
                    group_size=group_size,
                    rows=rows,
                    seed=seed_mode,
                    batch=dest_batch,
                    iters=iters,
                )
            )
        res = _launch(part, bank, out, variant, group_size, rows, seed_mode, 1, dest_batch)
        wp, wr, wz = _score(res, expect)
        log.append(
            dict(
                idx=len(log),
                variant=variant,
                group_size=group_size,
                rows=rows,
                seed=seed_mode,
                batch=dest_batch,
                iters="pcc_probe",
                pcc=f"{wp:.7f}",
                rel_rms=f"{wr:.6f}",
                zero_leak=f"{wz:g}",
            )
        )
    with open(HERE / "perf_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "variant",
                "group_size",
                "rows",
                "seed",
                "batch",
                "iters",
                "pcc",
                "rel_rms",
                "zero_leak",
            ],
        )
        w.writeheader()
        w.writerows(log)
    print(f"\nwrote {HERE / 'perf_manifest.csv'} ({len(log)} launches)")
