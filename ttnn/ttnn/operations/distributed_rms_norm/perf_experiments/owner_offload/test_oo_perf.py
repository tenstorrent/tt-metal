# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""owner_offload bake-off driver: the real op vs the experiment copy (drms_oo, OO_MODE 0/1/2), same session,
interleaved rounds. Metric as test_distributed_rms_norm_perf_traced.py: per call / per chip DEVICE KERNEL DURATION,
untraced last-launched / untraced slowest / traced slowest-chip-per-call (median of TRACED calls).

Correctness: every variant's output must be BIT-IDENTICAL to the real op's (pure ordering change) and PCC > 0.999 vs
torch.

    scripts/run_safe_pytest.sh --no-precompile --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/owner_offload/test_oo_perf.py -s

Env: OO_CASES ("640x1792"), OO_VARIANTS ("real,c1,c2"; cN = copy with DRMS_OO=N), OO_ROUNDS (3), OO_STUB (1: also
the exchange-stubbed timing), OO_UNTRACED (3), OO_TRACED (8), OO_ZONES ("": one zoned run per listed variant at the
end, DRMS_PERF_ZONES=1).
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as real_op, plan_summary
from drms_oo import distributed_rms_norm as copy_op

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_list = lambda name, default: [x for x in os.environ.get(name, default).split(",") if x]
CASES = [tuple(int(v) for v in c.split("x")) for c in _list("OO_CASES", "640x1792")]
VARIANTS = _list("OO_VARIANTS", "real,c1,c2")
ROUNDS = int(os.environ.get("OO_ROUNDS", "3"))
STUB = os.environ.get("OO_STUB", "1") != "0"
UNTRACED = int(os.environ.get("OO_UNTRACED", "3"))
TRACED = int(os.environ.get("OO_TRACED", "8"))
ZONES = _list("OO_ZONES", "")

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
    """(op callable, env) of a variant name: real | cN"""
    if name == "real":
        return real_op, {}
    return copy_op, {"DRMS_OO": name[1:]}


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
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_oo_perf(mesh_device):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    mesh = mesh_device
    G = 4
    for shape in CASES:
        M, W = shape
        torch.manual_seed(0)
        x = torch.randn((G, 1, 1, M, W), dtype=torch.float32).to(torch.bfloat16)
        g = torch.randn((G, 1, 1, 1, W), dtype=torch.float32).to(torch.bfloat16)
        put = lambda t: ttnn.from_torch(
            torch.cat(list(t), dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 3), mesh_shape=(1, 4)),
        )
        tx, tg = put(x), put(g)
        print(f"\nPLAN {shape} {plan_summary(tx, cluster_axis=1, gamma=tg, topology=ttnn.Topology.Ring, num_links=2)}")
        full = torch.cat(list(x.float()), dim=-1)
        ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), torch.cat(list(g.float().reshape(G, -1))), 1e-5)

        def runner(name):
            op, env = _variant(name)

            def run():
                with _Env(**env):
                    return op(tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2)

            return run

        gold = None
        for name in ["real"] + [v for v in VARIANTS if v != "real"]:
            out = runner(name)()
            got = torch.cat([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)], dim=-1)
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            if gold is None:
                gold = got
            exact = torch.equal(got, gold)
            print(f"CORRECT {M}x{W} {name}: pcc {pcc:.6f} bit-identical-to-real {exact}")
            assert pcc > 0.999
            assert exact, f"{name} differs from the real op"

        res = {}
        for rnd in range(ROUNDS):
            for name in VARIANTS:
                for stub in [False, True] if STUB else [False]:
                    op, env = _variant(name)
                    if stub:
                        env = dict(env, DRMS_ABLATE="exchange")
                    with _Env(**env):
                        r = _measure(mesh, runner(name))
                    key = (name, "stub" if stub else "op")
                    res.setdefault(key, []).append(r)
                    print(
                        f"PERF {M}x{W} r{rnd} {name:4s} {key[1]:4s}: untraced last {r[0]:6.2f} | untraced slowest "
                        f"{r[1]:6.2f} | traced slowest {r[2]:6.2f} (range {r[3]:.2f}-{r[4]:.2f}) us"
                    )
        for key, rs in res.items():
            med = lambda i: statistics.median(r[i] for r in rs)
            print(
                f"SUMMARY {M}x{W} {key[0]:4s} {key[1]:4s}: traced slowest med-of-rounds {med(2):6.2f} "
                f"[{', '.join(f'{r[2]:.2f}' for r in rs)}] | untraced last {med(0):6.2f} | untraced slowest {med(1):6.2f}"
            )
        for name in ZONES:
            op, env = _variant(name)
            with _Env(**dict(env, DRMS_PERF_ZONES="1")):
                _measure(mesh, runner(name))
            print(f"ZONES {M}x{W} {name}: profile_log_device.csv last run")
