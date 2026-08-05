# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""BENCH A driver: correctness gate (the ONLY pass/fail) + measured sweep.

    correctness:  scripts/run_safe_pytest.sh --run-all <this file> -k correctness
    perf:         scripts/run_safe_pytest.sh --profile <this file> -k perf_sweep
                  then: python3 <this dir>/read_perf.py

The perf test issues ONE generic_op launch per (config, iters) in a fixed order and writes that
order to perf_manifest.csv; read_perf.py joins it against the profiler's
DEVICE KERNEL DURATION [ns] column by launch order and reports (t_N - t_1) / (N - 1), which
cancels the fixed kernel-launch floor exactly.
"""

import csv
import importlib.util
import pathlib

import pytest

import ttnn

HERE = pathlib.Path(__file__).parent


def _load(name):
    # Load BY PATH, deliberately NOT as `ttnn.operations....`: ttnn/ttnn/operations/__init__.py
    # walk_packages()es and EXECUTES every reachable module at `import ttnn`.  See
    # perf_experiments/README.md.
    spec = importlib.util.spec_from_file_location(f"_cpt3_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases = _load("cases")
combine_bench = _load("combine_bench")

ITERS_HI = 9
EF, ET = combine_bench.BANK_EF, combine_bench.BANK_ET
NONE, COPY = combine_bench.SEED_NONE, combine_bench.SEED_COPY
SKIP, CFULL, RC = combine_bench.FIN_SKIP, combine_bench.FIN_C, combine_bench.FIN_RC

# BLOCK_ROWS: 1 makes the mechanism a no-op (expected FLAT, and FLAT IS IN-DOMAIN).
ROWS_SWEEP = (1, 2, 8, 16, 32)
# GROUP_SIZE: the op's real group sizes.  9 is ODD -> GATHER_SLOTS = 10 with a zero pad slot.
G_SWEEP = (4, 8, 9, 28, 32)
FOCUS_G, FOCUS_ROWS = 8, 8


def _fin_for(variant, rows):
    """The NARROWEST finalize scope that is CORRECT for the variant's stat-tile layout."""
    if variant in ("base_d22", "base_d19"):
        return SKIP  # a column-shaped stat: D17's even-parity <2,4> reaches column 0
    if variant in ("cand_recv", "cand_root"):
        if rows == 1:
            return SKIP  # one packed column == column 0, so D17's scope still covers it
        return CFULL if rows <= 16 else RC
    return SKIP  # no finalize in the sender / receiver variants


def _cfg(variant, group_size, rows, *, fin=None, bank=ET, bankdt="fp32", seed=NONE, batch=4):
    return (variant, group_size, rows, _fin_for(variant, rows) if fin is None else fin, bank, bankdt, seed, batch)


def _perf_configs():
    out = []
    # --- the headline domain sweep: BLOCK_ROWS x GROUP_SIZE, baseline vs candidate -----------
    for rows in ROWS_SWEEP:
        for gs in G_SWEEP:
            out.append(_cfg("base_d22", gs, rows))
            out.append(_cfg("cand_recv", gs, rows))
    # --- cross-round calibration: r2's pre-D22 baseline on THIS box --------------------------
    for rows in (1, 8, 32):
        out.append(_cfg("base_d19", FOCUS_G, rows))
    # --- the OTHER un-permute placement: on the root, serial ---------------------------------
    for rows in ROWS_SWEEP:
        out.append(_cfg("cand_root", FOCUS_G, rows))
    # --- the two sides the compact layout ADDS (sender pack, receiver un-pack) ---------------
    for rows in ROWS_SWEEP:
        out.append(_cfg("member_pack", FOCUS_G, rows))
        out.append(_cfg("recv_unpack", FOCUS_G, rows))
    # --- OPTION: what widening the finalize scope costs (the BLOCK_ROWS > 16 hazard) ---------
    for rows in (1, 2, 8, 16):
        out.append(_cfg("cand_recv", FOCUS_G, rows, fin=RC))
    out.append(_cfg("cand_recv", FOCUS_G, 8, fin=SKIP))  # measured only; WRONG above rows == 1
    # --- OPTION: bank spelling (EF vs E+transpose) and bank dtype (fp32 vs bf16) -------------
    for rows in (8, 32):
        for bank in (EF, ET):
            for dt in ("fp32", "bf16"):
                out.append(_cfg("recv_unpack", FOCUS_G, rows, bank=bank, bankdt=dt))
                out.append(_cfg("member_pack", FOCUS_G, rows, bank=bank, bankdt=dt))
    # --- OPTION: explicit DEST zero-seed instead of relying on the packer's ZEROACC ----------
    for rows in (8, 32):
        out.append(_cfg("member_pack", FOCUS_G, rows, seed=COPY))
        out.append(_cfg("recv_unpack", FOCUS_G, rows, seed=COPY))
    # --- OPTION: how much the un-pack's DEST batching is worth -------------------------------
    for rows in (8, 32):
        for b in (1, 2, 8):
            out.append(_cfg("recv_unpack", FOCUS_G, rows, batch=b))
            out.append(_cfg("cand_root", FOCUS_G, rows, batch=b))
    return out


def _correctness_configs():
    out = []
    for rows in ROWS_SWEEP:
        for gs in G_SWEEP:
            out.append(_cfg("base_d22", gs, rows))
            out.append(_cfg("cand_recv", gs, rows))
    for rows in ROWS_SWEEP:
        out.append(_cfg("cand_root", FOCUS_G, rows))
        for bank in (EF, ET):
            for dt in ("fp32", "bf16"):
                out.append(_cfg("member_pack", FOCUS_G, rows, bank=bank, bankdt=dt))
                out.append(_cfg("recv_unpack", FOCUS_G, rows, bank=bank, bankdt=dt))
    # the widened finalize must not change the answer on the lanes that matter
    for rows in ROWS_SWEEP:
        out.append(_cfg("cand_recv", FOCUS_G, rows, fin=RC))
    # the explicit DEST seed and the DEST batching must not change the answer either
    for rows in (8, 32):
        out.append(_cfg("member_pack", FOCUS_G, rows, seed=COPY))
        out.append(_cfg("recv_unpack", FOCUS_G, rows, seed=COPY))
        for b in (1, 2, 8):
            out.append(_cfg("recv_unpack", FOCUS_G, rows, batch=b))
            out.append(_cfg("cand_root", FOCUS_G, rows, batch=b))
    out.append(_cfg("base_d19", FOCUS_G, FOCUS_ROWS))
    return out


# The recorded REBASE HAZARD, measured rather than asserted: D17's shipped <2,4> even-parity
# finalize scope reaches only columns 0,2,..,14, so on a COMPACT tile it never scales or rsqrts
# the ODD rows' stats.  r2 measured pcc 0.9974 at BLOCK_ROWS = 2.  This is an xfail, not a skip:
# if it ever PASSES, the compact layout's lane invariant has changed and the guard must be re-read.
HAZARD_CONFIGS = [_cfg("cand_recv", FOCUS_G, r, fin=SKIP) for r in (2, 8)]


def _ids(cfgs):
    return [f"{v}_g{g}_r{r}_f{f}_bk{bk}{dt}_s{s}_b{b}" for v, g, r, f, bk, dt, s, b in cfgs]


def _alloc(device, variant, group_size, rows, bank_mode, bankdt):
    import torch

    part_t, bank_t, expect = cases.make_case(variant, group_size, rows, bank_mode)
    in_pages, bank_pages, out_pages = combine_bench.geometry(variant, group_size, rows, bank_mode)
    assert part_t.shape[0] == in_pages * cases.TILE, (variant, part_t.shape, in_pages)
    assert bank_t.shape[0] == bank_pages * cases.TILE

    def dev(t, pages, dtype=ttnn.float32):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=combine_bench.single_core_shard(pages),
        )

    part = dev(part_t, in_pages)
    # A one-hot tile is EXACTLY representable in bf16, so the bank dtype is a pure L1 lever.
    bank = dev(bank_t, bank_pages, ttnn.bfloat16 if bankdt == "bf16" else ttnn.float32)
    out = dev(torch.full((out_pages * cases.TILE, cases.TILE), -7.0, dtype=torch.float32), out_pages)
    return part, bank, out, expect


def _launch(part, bank, out, cfg, iters):
    variant, group_size, rows, fin, bank_mode, _dt, seed, batch = cfg
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
        fin=fin,
        bank_mode=bank_mode,
        seed=seed,
        dest_batch=batch,
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


@pytest.mark.parametrize("cfg", _correctness_configs(), ids=_ids(_correctness_configs()))
def test_correctness(device, cfg):
    """The only pass/fail in this experiment.  Perf is measured, never asserted."""
    variant, group_size, rows, fin, bank_mode, dt, seed, batch = cfg
    part, bank, out, expect = _alloc(device, variant, group_size, rows, bank_mode, dt)
    res = _launch(part, bank, out, cfg, 2)
    wp, wr, wz = _score(res, expect)
    print(f"\n{cfg}: pcc={wp:.7f} rel_rms={wr:.6f} zero_leak={wz:g}")
    assert wz == 0.0, f"positions that must be exactly 0 hold {wz}"
    assert wp > 0.9995, f"pcc {wp}"
    assert wr <= 0.04, f"rel_rms {wr}"


@pytest.mark.parametrize("cfg", HAZARD_CONFIGS, ids=_ids(HAZARD_CONFIGS))
def test_finalize_scope_hazard(device, cfg):
    """D17's <2,4> scope on a COMPACT tile is WRONG from BLOCK_ROWS = 2 -- kept as a live number."""
    variant, group_size, rows, fin, bank_mode, dt, seed, batch = cfg
    part, bank, out, expect = _alloc(device, variant, group_size, rows, bank_mode, dt)
    res = _launch(part, bank, out, cfg, 2)
    wp, wr, wz = _score(res, expect)
    print(f"\nHAZARD {cfg}: pcc={wp:.7f} rel_rms={wr:.6f}")
    assert wp <= 0.9995 or wr > 0.04, (
        f"D17's <2,4> scope UNEXPECTEDLY passed on a compact tile (pcc {wp}, rel_rms {wr}) -- "
        "re-read the lane invariant before trusting the narrow scope"
    )


def test_perf_sweep(device):
    """One launch per (config, iters) in a deterministic order; the manifest records that order."""
    log = []
    for cfg in _perf_configs():
        variant, group_size, rows, fin, bank_mode, dt, seed, batch = cfg
        part, bank, out, expect = _alloc(device, variant, group_size, rows, bank_mode, dt)
        common = dict(
            variant=variant,
            group_size=group_size,
            rows=rows,
            fin=fin,
            bank=bank_mode,
            bankdt=dt,
            seed=seed,
            batch=batch,
        )
        for iters in (1, ITERS_HI):
            _launch(part, bank, out, cfg, iters)
            ttnn.synchronize_device(device)
            log.append(dict(idx=len(log), iters=iters, **common))
        res = _launch(part, bank, out, cfg, 1)
        wp, wr, wz = _score(res, expect)
        log.append(
            dict(
                idx=len(log),
                iters="pcc_probe",
                pcc=f"{wp:.7f}",
                rel_rms=f"{wr:.6f}",
                zero_leak=f"{wz:g}",
                **common,
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
                "fin",
                "bank",
                "bankdt",
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
