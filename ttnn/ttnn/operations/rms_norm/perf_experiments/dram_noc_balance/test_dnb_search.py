# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT dram_noc_balance — in-session coordinate-descent search over per-grid-column / per-core NoC masks.

Exploration tool only (the reported numbers come from test_dram_noc_balance.py's one-run-per-variant CSV A/B).
Each evaluation dispatches the op through policy "table" and reads DEVICE KERNEL DURATION in-process
(ttnn.ReadDeviceProfiler; this module sets the TT_METAL_DEVICE_PROFILER / MID_RUN_DUMP env itself).
State = per virtual grid column (DNB_SEARCH_UNIT=col) or per core (=core) an option pair indexing
dnb_program_descriptor.DNB_RD_OPTS / DNB_WR_OPTS. Objective = mean over DNB_SEARCH_CASES of (min-of-reps ns / that
case's baseline ns).

    DNB_SEARCH_CASES=focus,i8192x1024 scripts/run_safe_pytest.sh --no-precompile <this file> --import-mode=prepend -s
"""

import itertools
import json
import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", "1")

import pytest

torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnb_program_descriptor as dpd  # noqa: E402
from test_dram_noc_balance import CASES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_KEY = "DEVICE KERNEL DURATION [ns]"
NR, NW = len(dpd.DNB_RD_OPTS), len(dpd.DNB_WR_OPTS)


def _device_ns(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


@pytest.mark.timeout(3400)
def test_dnb_search(device, monkeypatch):
    cases = os.environ.get("DNB_SEARCH_CASES", "focus").split(",")
    unit = os.environ.get("DNB_SEARCH_UNIT", "col")
    reps = int(os.environ.get("DNB_SEARCH_REPS", "3"))
    sweeps = int(os.environ.get("DNB_SEARCH_SWEEPS", "2"))
    tag = os.environ.get("DNB_SEARCH_TAG", "_".join(cases) + "_" + unit)
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    monkeypatch.setenv("DNB_POLICY", "table")
    setups = []
    for case in cases:
        rows, hidden, ml, shard, g_layout, fp32_dest, x_layout = CASES[case]
        torch.manual_seed(0)
        x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
        g = torch.randn(hidden).to(torch.bfloat16)
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=x_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=g_layout, device=device)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, device, tx.memory_config()
        )
        setups.append((tx, tg, cfg, out))
    grid = device.compute_with_storage_grid_size()
    cores = [device.worker_core_from_logical_core(ttnn.CoreCoord(i, j)) for j in range(grid.y) for i in range(grid.x)]
    cores = [(c.x, c.y) for c in cores]
    units = sorted({c[0] for c in cores}) if unit == "col" else cores

    def members(u):
        return [c for c in cores if c[0] == u] if unit == "col" else [u]

    def raw(state):
        dpd.DNB_TABLE.clear()
        for u, (r, w) in state.items():
            for c in members(u):
                dpd.DNB_TABLE[c] = (dpd.DNB_RD_OPTS[r], dpd.DNB_WR_OPTS[w])
        res = []
        for tx, tg, cfg, out in setups:
            pd = dpd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
            _device_ns(device)  # drain
            for _ in range(reps):
                ttnn.generic_op([tx, tg, out], pd)
            ns = _device_ns(device)
            res.append(min(ns))
        return res

    zero = {u: (0, 0) for u in units}
    base = [min(a, b) for a, b in zip(raw(zero), raw(zero))]

    def evaluate(state):
        return sum(t / b for t, b in zip(raw(state), base)) / len(base)

    state = dict(zero)
    init = os.environ.get("DNB_SEARCH_INIT")
    if init:
        for k, v in json.load(open(os.path.join(HERE, init))).items():
            u = int(k) if "," not in k else tuple(int(t) for t in k.split(","))
            if unit == "col" and isinstance(u, int) and u in state:
                state[u] = tuple(v)
            elif unit == "core":
                for c in cores:
                    if (isinstance(u, int) and c[0] == u) or u == c:
                        state[c] = tuple(v)
    best = max(evaluate(state), evaluate(state))
    log = open(os.path.join(HERE, f"dnb_search_{tag}.log"), "a")
    msg = f"DNBS start cases={cases} base_ns={base} start_obj={best:.4f}"
    print(msg)
    log.write(msg + "\n")
    for sweep in range(sweeps):
        improved = False
        for u in units:
            cur = state[u]
            for r, w in itertools.product(range(NR), range(NW)):
                if (r, w) == cur:
                    continue
                trial = dict(state)
                trial[u] = (r, w)
                if evaluate(trial) < best * 0.995:
                    t = max(evaluate(trial), evaluate(trial))  # confirm twice (noise guard)
                    if t < best * 0.997:
                        best, state, improved = t, trial, True
                        msg = f"DNBS sweep {sweep} unit {u} -> {(r, w)} obj={best:.4f}"
                        print(msg)
                        log.write(msg + "\n")
                        log.flush()
        if not improved:
            break
    key = (lambda u: str(u)) if unit == "col" else (lambda u: f"{u[0]},{u[1]}")
    final = {key(u): list(v) for u, v in state.items() if v != (0, 0)}
    with open(os.path.join(HERE, f"dnb_search_{tag}.json"), "w") as fh:
        json.dump(final, fh, indent=1)
    fin = [raw(state) for _ in range(3)]
    bas = [raw(zero) for _ in range(3)]
    msg = f"DNBS final {final}\n  final_ns={fin}\n  base_ns={bas}"
    print(msg)
    log.write(msg + "\n")


@pytest.mark.timeout(3400)
def test_dnb_eval(device, monkeypatch):
    """Quick in-session screen: DNB_EVAL_POLICIES (';'-separated grammar / json: policies) on DNB_SEARCH_CASES,
    DNB_EVAL_ROUNDS rounds interleaved with base, min-of-reps per (round, policy, case); prints the median ratio."""
    cases = os.environ.get("DNB_SEARCH_CASES", "focus").split(",")
    reps = int(os.environ.get("DNB_SEARCH_REPS", "3"))
    rounds = int(os.environ.get("DNB_EVAL_ROUNDS", "3"))
    policies = ["base"] + [p for p in os.environ.get("DNB_EVAL_POLICIES", "").split(";") if p]
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    setups = []
    for case in cases:
        rows, hidden, ml, shard, g_layout, fp32_dest, x_layout = CASES[case]
        torch.manual_seed(0)
        x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
        g = torch.randn(hidden).to(torch.bfloat16)
        mc = ttnn.DRAM_MEMORY_CONFIG
        if shard is not None:
            from eval.sharding import shard_config

            mc = shard_config(shard[0], shard[1], ml, layout=x_layout, dtype=ttnn.bfloat16, device=device)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=x_layout, device=device, memory_config=mc)
        tg = None
        if g_layout is not None:
            tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=g_layout, device=device)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, device, tx.memory_config()
        )
        setups.append((tx, tg, cfg, out))
    res = {p: {c: [] for c in cases} for p in policies}
    for _ in range(rounds):
        for p in policies:
            monkeypatch.setenv("DNB_POLICY", p)
            for case, (tx, tg, cfg, out) in zip(cases, setups):
                pd = dpd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
                io = [tx] + ([tg] if tg is not None else []) + [out]
                _device_ns(device)
                for _ in range(reps):
                    ttnn.generic_op(io, pd)
                res[p][case].append(min(_device_ns(device)))
    log = open(os.path.join(HERE, "dnb_eval.log"), "a")
    for p in policies:
        cells = []
        for c in cases:
            med = sorted(res[p][c])[len(res[p][c]) // 2]
            bmed = sorted(res["base"][c])[len(res["base"][c]) // 2]
            cells.append(f"{c}={med:.0f}({bmed / med:.3f}x)")
        msg = f"DNBE {p:50s} " + " ".join(cells)
        print(msg)
        log.write(msg + "\n")


@pytest.mark.timeout(3400)
def test_dnb_ded_search(device, monkeypatch):
    """Greedy search over which units (grid columns, DNB_SEARCH_UNIT=col; or cores, =core) run the swapped DEDICATED
    kernels (policy ded:<union of X / P sets>). Objective = mean over DNB_SEARCH_CASES of min-of-reps ns / base ns."""
    cases = os.environ.get("DNB_SEARCH_CASES", "focus").split(",")
    unit = os.environ.get("DNB_SEARCH_UNIT", "col")
    reps = int(os.environ.get("DNB_SEARCH_REPS", "3"))
    sweeps = int(os.environ.get("DNB_SEARCH_SWEEPS", "3"))
    tag = os.environ.get("DNB_SEARCH_TAG", "ded_" + "_".join(cases) + "_" + unit)
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    setups = []
    for case in cases:
        rows, hidden, ml, shard, g_layout, fp32_dest, x_layout = CASES[case]
        torch.manual_seed(0)
        x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
        g = torch.randn(hidden).to(torch.bfloat16)
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=x_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=g_layout, device=device)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, device, tx.memory_config()
        )
        setups.append((tx, tg, cfg, out))
    grid = device.compute_with_storage_grid_size()
    cores = [device.worker_core_from_logical_core(ttnn.CoreCoord(i, j)) for j in range(grid.y) for i in range(grid.x)]
    cores = [(c.x, c.y) for c in cores]
    units = sorted({c[0] for c in cores}) if unit == "col" else cores

    def policy(on):
        if not on:
            return "ded:N"
        return "ded:" + "|".join(f"X{u}-{u}" if unit == "col" else f"P{u[0]}.{u[1]}" for u in sorted(on))

    def raw(on):
        monkeypatch.setenv("DNB_POLICY", policy(on))
        res = []
        for tx, tg, cfg, out in setups:
            pd = dpd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
            _device_ns(device)
            for _ in range(reps):
                ttnn.generic_op([tx, tg, out], pd)
            res.append(min(_device_ns(device)))
        return res

    base = [min(a, b) for a, b in zip(raw(set()), raw(set()))]

    def evaluate(on):
        return sum(t / b for t, b in zip(raw(on), base)) / len(base)

    on = set()
    init = os.environ.get("DNB_SEARCH_INIT_SET")
    if init:
        for tok in init.split(","):
            on.add(int(tok) if unit == "col" else tuple(int(t) for t in tok.split(".")))
    best = max(evaluate(on), evaluate(on))
    log = open(os.path.join(HERE, f"dnb_search_{tag}.log"), "a")
    msg = f"DNBD start cases={cases} base_ns={base} init={sorted(on)} obj={best:.4f}"
    print(msg)
    log.write(msg + "\n")
    for sweep in range(sweeps):
        improved = False
        for u in units:
            trial = set(on) ^ {u}
            if evaluate(trial) < best * 0.995:
                t = max(evaluate(trial), evaluate(trial))
                if t < best * 0.997:
                    best, on, improved = t, trial, True
                    msg = f"DNBD sweep {sweep} toggle {u} -> on={sorted(on)} obj={best:.4f}"
                    print(msg)
                    log.write(msg + "\n")
                    log.flush()
        if not improved:
            break
    fin = [raw(on) for _ in range(3)]
    bas = [raw(set()) for _ in range(3)]
    msg = f"DNBD final policy={policy(on)}\n  final_ns={fin}\n  base_ns={bas}"
    print(msg)
    log.write(msg + "\n")
