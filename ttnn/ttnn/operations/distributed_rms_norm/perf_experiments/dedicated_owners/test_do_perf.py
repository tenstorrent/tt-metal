# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""dedicated_owners bake-off driver: the real op vs the experiment copy (drms_do, DRMS_DO = 0 / 1 / 2), same session,
interleaved rounds. Metric as test_distributed_rms_norm_perf_traced.py: per call / per chip DEVICE KERNEL DURATION —
untraced last-launched (min over chips) / untraced slowest / traced slowest-chip-per-call (median of TRACED calls).

Correctness: every variant's output must be BIT-IDENTICAL to the real op's (same arithmetic, same fold order) and
PCC > 0.9999 vs torch.

    scripts/run_safe_pytest.sh --no-precompile --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/dedicated_owners/test_do_perf.py -s \
        --import-mode=prepend

Env: DO_CASES (case ids below, default "D3-focus"), DO_VARIANTS ("real,d1"; dN = copy with DRMS_DO=N), DO_ROUNDS (3),
DO_STUB (1: also the exchange-stubbed timing), DO_UNTRACED (3), DO_TRACED (8), DO_ZONES ("": one zoned run per
listed variant at the end, DRMS_PERF_ZONES=1; the profiler CSV is copied to zones/<case>_<variant>.csv),
DO_TOPO (Ring | Linear), DO_LINKS (2 | 1), DO_AXIS (1: 1 x 4 mesh | 0: 4 x 1 mesh).
"""

import importlib.util as _ilu
import os
import pathlib as _pl
import shutil
import statistics
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import ttnn

HERE = _pl.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as real_op
from drms_do import distributed_rms_norm as copy_op, plan_summary as copy_plan_summary
import drms_do.distributed_rms_norm_program_descriptor as dpd

_REPO = HERE.parents[5]
_UT = _REPO / "tests/ttnn/unit_tests/operations/distributed_rms_norm"
_spec = _ilu.spec_from_file_location("_drms_accept", _UT / "test_distributed_rms_norm.py")
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

ML = ttnn.TensorMemoryLayout
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
ALL_CASES = {
    "D3-focus": ((640, 1792), ML.INTERLEAVED, TILE),
    "D3-rmg": ((640, 1792), ML.INTERLEAVED, RM),
    "D3-nog": ((640, 1792), ML.INTERLEAVED, None),
    "D3-decode": ((32, 1792), ML.INTERLEAVED, TILE),
    "D3-2048": ((2048, 1280), ML.INTERLEAVED, TILE),
    "D1": ((256, 96), ML.INTERLEAVED, TILE),
    "D1-HS": ((1024, 512), ML.HEIGHT_SHARDED, TILE),
    "D3-WS": ((64, 2048), ML.WIDTH_SHARDED, TILE),
    "D3-WS-nog": ((64, 2048), ML.WIDTH_SHARDED, None),
}
_list = lambda name, default: [x for x in os.environ.get(name, default).split(",") if x]
CASES = _list("DO_CASES", "D3-focus")
VARIANTS = _list("DO_VARIANTS", "real,d1")
ROUNDS = int(os.environ.get("DO_ROUNDS", "3"))
STUB = os.environ.get("DO_STUB", "1") != "0"
UNTRACED = int(os.environ.get("DO_UNTRACED", "3"))
TRACED = int(os.environ.get("DO_TRACED", "8"))
ZONES = _list("DO_ZONES", "")
TOPO = getattr(ttnn.Topology, os.environ.get("DO_TOPO", "Ring"))
LINKS = int(os.environ.get("DO_LINKS", "2"))
AXIS = int(os.environ.get("DO_AXIS", "1"))
MESH = (1, 4) if AXIS == 1 else (4, 1)
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")


def _per_chip_ns(mesh):
    ttnn.ReadDeviceProfiler(mesh)
    out = {}
    for dev, programs in (ttnn.get_latest_programs_perf_data() or {}).items():
        out[dev] = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
    return out


def _measure(mesh, run):
    run()
    ttnn.synchronize_device(mesh)
    _per_chip_ns(mesh)
    last, slow = [], []
    for _ in range(UNTRACED):
        run()
        ttnn.synchronize_device(mesh)
        per = [v[-1] for v in _per_chip_ns(mesh).values() if v]
        last.append(min(per))
        slow.append(max(per))
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(TRACED):
        run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh, tid)
    ttnn.synchronize_device(mesh)
    per = [v[-TRACED:] for v in _per_chip_ns(mesh).values() if len(v) >= TRACED]
    traced = [max(chip[i] for chip in per) for i in range(TRACED)]
    us = lambda v: statistics.median(v) / 1000
    return us(last), us(slow), us(traced), min(traced) / 1000, max(traced) / 1000


class _Env:
    def __init__(self, **kv):
        self.kv = {k: v for k, v in kv.items() if v is not None}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        os.environ.update(self.kv)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _variant(name):
    if name == "real":
        return real_op, {}
    return copy_op, {"DRMS_DO": name[1:]}


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": production_router_config(),
            "trace_region_size": 1 << 23,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], ids=[f"mesh{MESH[0]}x{MESH[1]}"], indirect=True)
def test_do_perf(mesh_device):
    import torch

    mesh = mesh_device
    summary = []
    for cid in CASES:
        (M, W), ml, gl = ALL_CASES[cid]
        shape = (1, 1, M, W)
        torch.manual_seed(0)
        x = _acc._stacked(MESH, shape, torch.bfloat16)
        mc = ttnn.DRAM_MEMORY_CONFIG
        if ml != ML.INTERLEAVED:
            mc = auto_shard_config(list(shape), ml, layout=TILE, dtype=ttnn.bfloat16, device=mesh)
        tx = _acc._to_mesh(x, mesh, dtype=ttnn.bfloat16, layout=TILE, memory_config=mc)
        gamma = tg = None
        if gl is not None:
            gamma = _acc._stacked(MESH, (1, 1, 1, W), torch.bfloat16)
            tg = _acc._gamma_to_mesh(gamma, mesh, dtype=ttnn.bfloat16, layout=gl)
        expected = _acc._reference(x, gamma, AXIS, 1e-5)
        out_mc = mc if ml != ML.INTERLEAVED else None
        print(f"\nPLAN {cid} {copy_plan_summary(tx, cluster_axis=AXIS, gamma=tg, topology=TOPO, num_links=LINKS)}")

        def runner(name):
            op, env = _variant(name)

            def run():
                with _Env(**env):
                    return op(
                        tx,
                        cluster_axis=AXIS,
                        gamma=tg,
                        epsilon=1e-5,
                        topology=TOPO,
                        num_links=LINKS,
                        memory_config=out_mc,
                    )

            return run

        # poison: a call on different data first, so a read of a stale landing record (the previous call's) shows
        x2 = _acc._stacked(MESH, shape, torch.bfloat16, scale=3.0)
        tx2 = _acc._to_mesh(x2, mesh, dtype=ttnn.bfloat16, layout=TILE, memory_config=mc)

        def poisoned(name):
            op, env = _variant(name)
            with _Env(**env):
                op(tx2, cluster_axis=AXIS, gamma=tg, epsilon=1e-5, topology=TOPO, num_links=LINKS, memory_config=out_mc)
            return runner(name)()

        gold = None
        for name in ["real"] + [v for v in VARIANTS if v != "real"]:
            out = poisoned(name)
            devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)]
            pcc = min(_acc._pcc(d, expected[divmod(i, MESH[1])]) for i, d in enumerate(devs))
            got = torch.stack(devs)
            if gold is None:
                gold = got
            exact = torch.equal(got, gold)
            print(f"CORRECT {cid} {name}: pcc {pcc:.7f} bit-identical-to-real {exact}")
            if name != "real":
                print(f"SELECTION {cid} {name}: {dict(dpd._LAST_SELECTION)}")
            assert pcc > 0.9999, (cid, name, pcc)
            assert exact, f"{cid}: {name} differs from the real op"

        res = {}
        for rnd in range(ROUNDS):
            order = VARIANTS if rnd % 2 == 0 else list(reversed(VARIANTS))
            for name in order:
                for stub in [False, True] if STUB else [False]:
                    op, env = _variant(name)
                    if stub:
                        env = dict(env, DRMS_ABLATE="exchange")
                    with _Env(**env):
                        r = _measure(mesh, runner(name))
                    key = (name, "stub" if stub else "op")
                    res.setdefault(key, []).append(r)
                    print(
                        f"PERF {cid} r{rnd} {name:4s} {key[1]:4s}: untraced last {r[0]:6.2f} | untraced slowest "
                        f"{r[1]:6.2f} | traced slowest {r[2]:6.2f} (range {r[3]:.2f}-{r[4]:.2f}) us"
                    )
        for key, rs in res.items():
            med = lambda i: statistics.median(r[i] for r in rs)
            line = (
                f"SUMMARY {cid:10s} {key[0]:4s} {key[1]:4s}: traced slowest med {med(2):6.2f} "
                f"[{', '.join(f'{r[2]:.2f}' for r in rs)}] | untraced last {med(0):6.2f} "
                f"[{', '.join(f'{r[0]:.2f}' for r in rs)}] | untraced slowest {med(1):6.2f}"
            )
            print(line)
            summary.append(line)
        for name in ZONES:
            op, env = _variant(name)
            with _Env(**dict(env, DRMS_PERF_ZONES="1")):
                _measure(mesh, runner(name))
            src = _REPO / "generated/profiler/.logs/profile_log_device.csv"
            dst = HERE / "zones" / f"{cid}_{name}.csv"
            dst.parent.mkdir(exist_ok=True)
            if src.exists():
                shutil.copy(src, dst)
            print(f"ZONES {cid} {name}: {dst}")
    print("\n" + "\n".join(summary))
