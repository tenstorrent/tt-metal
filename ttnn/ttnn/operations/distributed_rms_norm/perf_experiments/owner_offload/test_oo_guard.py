# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""owner_offload domain sweep: the perf-guard config set (test_distributed_rms_norm_perf_guard.py CASES, incl. the
forced L1-ladder rungs) — real op vs the copy (OO_MODE N), same session, interleaved rounds. Every variant's output must
be bit-identical to the real op's; PCC > 0.999 vs torch.

    scripts/run_safe_pytest.sh --no-precompile --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/owner_offload/test_oo_guard.py -s --import-mode=prepend

Env: OO_GUARD_CASES (ids), OO_VARIANTS ("real,c1,c2"), OO_ROUNDS (3), OO_UNTRACED (2), OO_TRACED (8),
DRMS_GUARD_TOPO / DRMS_GUARD_LINKS / DRMS_GUARD_AXIS as the guard.
"""

import importlib.util as _ilu
import os
import pathlib as _pl
import statistics
import sys
from dataclasses import replace

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd_real
import drms_oo.distributed_rms_norm_program_descriptor as pd_copy
from drms_oo import distributed_rms_norm as copy_op
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as real_op, plan_summary

_UT = _pl.Path(__file__).resolve().parents[6] / "tests/ttnn/unit_tests/operations/distributed_rms_norm"
_spec = _ilu.spec_from_file_location("_drms_accept", _UT / "test_distributed_rms_norm.py")
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
_spec = _ilu.spec_from_file_location("_oo_perf", _pl.Path(__file__).with_name("test_oo_perf.py"))
_pf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pf)

ML = ttnn.TensorMemoryLayout
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
CASES = [
    ("D3-focus", (640, 1792), ML.INTERLEAVED, TILE, None),
    ("D3-rmg", (640, 1792), ML.INTERLEAVED, RM, None),
    ("D3-nog", (640, 1792), ML.INTERLEAVED, None, None),
    ("D3-decode", (32, 1792), ML.INTERLEAVED, TILE, None),
    ("D3-2048", (2048, 1280), ML.INTERLEAVED, TILE, None),
    ("D1", (256, 96), ML.INTERLEAVED, TILE, None),
    ("D1-HS", (1024, 512), ML.HEIGHT_SHARDED, TILE, None),
    ("D3-WS", (64, 2048), ML.WIDTH_SHARDED, TILE, None),
    ("D3-WS-nog", (64, 2048), ML.WIDTH_SHARDED, None, None),
    ("rung-passes", (640, 1792), ML.INTERLEAVED, TILE, dict(PW=1, x_depth=2)),
    ("rung-chunk-stream", (640, 1792), ML.INTERLEAVED, TILE, dict(gamma_mode="CHUNK", gamma_stream=True)),
    ("rung-reread", (640, 1792), ML.INTERLEAVED, TILE, dict(gamma_mode="CHUNK", sq_rows=1, reread=True)),
]
_sel = [c for c in os.environ.get("OO_GUARD_CASES", "").split(",") if c]
if _sel:
    CASES = [c for c in CASES if c[0] in _sel]
VARIANTS = [v for v in os.environ.get("OO_VARIANTS", "real,c1,c2").split(",") if v]
ROUNDS = int(os.environ.get("OO_ROUNDS", "3"))
_pf.UNTRACED = int(os.environ.get("OO_UNTRACED", "2"))
TOPO = getattr(ttnn.Topology, os.environ.get("DRMS_GUARD_TOPO", "Ring"))
LINKS = int(os.environ.get("DRMS_GUARD_LINKS", "2"))
AXIS = int(os.environ.get("DRMS_GUARD_AXIS", "1"))
MESH = (1, 4) if AXIS == 1 else (4, 1)

pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")


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
def test_oo_guard(mesh_device, monkeypatch):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    mesh = mesh_device
    lines = []
    for cid, (M, W), ml, gl, opt in CASES:
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

        def runner(name):
            op = real_op if name == "real" else copy_op
            env = {} if name == "real" else {"DRMS_OO": name[1:]}

            def run():
                with _pf._Env(**env):
                    return op(
                        tx,
                        cluster_axis=AXIS,
                        gamma=tg,
                        epsilon=1e-5,
                        topology=TOPO,
                        num_links=LINKS,
                        memory_config=out_mc,
                    )

            return run, env

        with monkeypatch.context() as mp:
            if opt is not None:
                for pd in (pd_real, pd_copy):
                    o = dict(opt)
                    if o.get("gamma_mode") == "CHUNK":
                        o["gamma_mode"] = pd.GAMMA_CHUNK
                    real_ff = pd.first_fit
                    mp.setattr(
                        pd,
                        "first_fit",
                        (lambda rf, oo: (lambda base, sz, budget: (0, replace(rf(base, sz, budget)[1], **oo))))(
                            real_ff, o
                        ),
                    )
            print(f"\nPLAN {cid} {plan_summary(tx, cluster_axis=AXIS, gamma=tg, num_links=LINKS)}")
            gold = None
            for name in ["real"] + [v for v in VARIANTS if v != "real"]:
                out = runner(name)[0]()
                devs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)]
                pcc = min(_acc._pcc(d, expected[divmod(i, MESH[1])]) for i, d in enumerate(devs))
                got = torch.stack(devs)
                gold = got if gold is None else gold
                exact = torch.equal(got, gold)
                print(f"CORRECT {cid} {name}: pcc {pcc:.6f} bit-identical-to-real {exact}")
                assert pcc > 0.999, (cid, name, pcc)
                assert exact, (cid, name)
            res = {}
            for _ in range(ROUNDS):
                for name in VARIANTS:
                    run, env = runner(name)
                    for stub in (False, True):
                        e = dict(env, DRMS_ABLATE="exchange") if stub else env
                        with _pf._Env(**e):
                            r = _pf._measure(mesh, run)
                        res.setdefault((name, stub), []).append(r[2])
            med = {k: statistics.median(v) for k, v in res.items()}
            line = f"GUARD {cid:18s} " + " | ".join(
                f"{n} op {med[(n, False)]:6.2f} stub {med[(n, True)]:6.2f}" for n in VARIANTS
            )
            print(line)
            for k, v in res.items():
                print(f"  RAW {cid} {k[0]} {'stub' if k[1] else 'op'}: {', '.join(f'{x:.2f}' for x in v)}")
            lines.append(line)
    print("\n" + "\n".join(lines))
