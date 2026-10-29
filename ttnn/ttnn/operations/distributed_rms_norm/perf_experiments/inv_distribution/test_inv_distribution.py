# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_experiments/inv_distribution: same-session interleaved A/B of the inv distribution hop (measurement driver).

Variants (INV_VARIANTS, ';'-separated): "base" = the real op (ttnn.operations.distributed_rms_norm), anything else =
this dir's copy (drms_copy/) with DRMS_INV_VARIANT=<the variant's comma list> ("copy" = the copy, no variant).
Every variant's output is checked (PCC vs torch > 0.999 and bit-exact vs the base op's output: the arithmetic is
unchanged, only the transport / layout of the inv record).

    INV_CASES=D3-focus INV_VARIANTS="base;mcast" INV_ROUNDS=3 scripts/run_safe_pytest.sh --no-precompile --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/inv_distribution/test_inv_distribution.py -s

Metric: traced slowest chip per call (median of DRMS_PT_TRACED=8 calls), op and exchange-stubbed.
"""

import importlib.util as _ilu
import os
import pathlib as _pl
import statistics
from dataclasses import replace

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import ttnn

import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd_base
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as op_base, plan_summary

import ttnn.operations.distributed_rms_norm.perf_experiments.inv_distribution.drms_copy as cp
import ttnn.operations.distributed_rms_norm.perf_experiments.inv_distribution.drms_copy.distributed_rms_norm_program_descriptor as pd_copy

_UT = _pl.Path(__file__).resolve().parents[6] / "tests/ttnn/unit_tests/operations/distributed_rms_norm"
_spec = _ilu.spec_from_file_location("_drms_traced", _UT / "test_distributed_rms_norm_perf_traced.py")
_tr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tr)
_spec = _ilu.spec_from_file_location("_drms_accept", _UT / "test_distributed_rms_norm.py")
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

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
    ("rung-passes", (640, 1792), ML.INTERLEAVED, TILE, dict(PW=1, x_depth=2)),
    ("rung-chunk-stream", (640, 1792), ML.INTERLEAVED, TILE, None),
    ("rung-reread", (640, 1792), ML.INTERLEAVED, TILE, None),
]
_sel = [c for c in os.environ.get("INV_CASES", "D3-focus").split(",") if c]
CASES = [c for c in CASES if c[0] in _sel]
VARIANTS = [v for v in os.environ.get("INV_VARIANTS", "base;mcast").split(";") if v]
ROUNDS = int(os.environ.get("INV_ROUNDS", "3"))
STUB = os.environ.get("INV_STUB", "1") == "1"
DTYPE = getattr(ttnn, os.environ.get("INV_DTYPE", "bfloat16"))
FP32_ACC = os.environ.get("INV_FP32_ACC", "1") == "1"

pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")
TOPO = getattr(ttnn.Topology, os.environ.get("INV_TOPO", "Ring"))
LINKS = int(os.environ.get("INV_LINKS", "2"))
AXIS = int(os.environ.get("INV_AXIS", "1"))
MESH = (1, 4) if AXIS == 1 else (4, 1)


def _opt(cid):
    if cid == "rung-chunk-stream":
        return dict(gamma_mode=pd_base.GAMMA_CHUNK, gamma_stream=True)
    if cid == "rung-reread":
        return dict(gamma_mode=pd_base.GAMMA_CHUNK, sq_rows=1, reread=True)
    return None


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
def test_inv_distribution(mesh_device, monkeypatch):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    mesh = mesh_device
    summary = []
    from ttnn.operations.distributed_rms_norm import default_compute_kernel_config

    ckc = default_compute_kernel_config()
    ckc.fp32_dest_acc_en = FP32_ACC
    for cid, (M, W), ml, gl, opt in CASES:
        opt = opt or _opt(cid)
        shape = (1, 1, M, W)
        tdt = torch.float32 if DTYPE == ttnn.float32 else torch.bfloat16
        x = _acc._stacked(MESH, shape, tdt)
        mc = ttnn.DRAM_MEMORY_CONFIG
        if ml != ML.INTERLEAVED:
            mc = auto_shard_config(list(shape), ml, layout=TILE, dtype=DTYPE, device=mesh)
        tx = _acc._to_mesh(x, mesh, dtype=DTYPE, layout=TILE, memory_config=mc)
        gamma = tg = None
        if gl is not None:
            gamma = _acc._stacked(MESH, (1, 1, 1, W), tdt)
            tg = _acc._gamma_to_mesh(gamma, mesh, dtype=DTYPE, layout=gl)
        expected = _acc._reference(x, gamma, AXIS, 1e-5)
        out_mc = mc if ml != ML.INTERLEAVED else None

        def runner(variant):
            fn = op_base if variant == "base" else cp.distributed_rms_norm
            env = "" if variant in ("base", "copy") else variant

            def run():
                os.environ["DRMS_INV_VARIANT"] = env
                try:
                    return fn(
                        tx,
                        cluster_axis=AXIS,
                        gamma=tg,
                        epsilon=1e-5,
                        topology=TOPO,
                        num_links=LINKS,
                        memory_config=out_mc,
                        compute_kernel_config=ckc,
                    )
                finally:
                    os.environ.pop("DRMS_INV_VARIANT", None)

            return run

        with monkeypatch.context() as mp:
            if opt is not None:
                for pd in (pd_base, pd_copy):
                    real = pd.first_fit
                    mp.setattr(
                        pd,
                        "first_fit",
                        (lambda r: lambda base, sz, budget: (0, replace(r(base, sz, budget)[1], **opt)))(real),
                    )
            print(f"\nPLAN {cid} {plan_summary(tx, cluster_axis=AXIS, gamma=tg, num_links=LINKS)}")
            ref_out = None
            pccs = {}
            for v in VARIANTS:
                out = runner(v)()
                got = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out)]
                pcc = min(_acc._pcc(g.float(), expected[divmod(i, MESH[1])]) for i, g in enumerate(got))
                if ref_out is None:
                    ref_out = got
                    exact = True
                else:
                    exact = all(torch.equal(a, b) for a, b in zip(got, ref_out))
                pccs[v] = (pcc, exact)
                print(f"CHECK {cid} {v:24s} pcc {pcc:.6f} bit-exact-vs-{VARIANTS[0]} {exact}")
                assert pcc > 0.999, (cid, v, pcc)
                assert exact, (cid, v, "not bit-exact vs", VARIANTS[0])
            res = {v: {"op": [], "stub": []} for v in VARIANTS}
            for rnd in range(ROUNDS):
                order = VARIANTS if rnd % 2 == 0 else VARIANTS[::-1]
                for v in order:
                    run = runner(v)
                    res[v]["op"].append(_tr._measure(mesh, run)[2])
                    if STUB:
                        os.environ["DRMS_ABLATE"] = "exchange"
                        try:
                            res[v]["stub"].append(_tr._measure(mesh, run)[2])
                        finally:
                            os.environ.pop("DRMS_ABLATE", None)
                    print(
                        f"ROUND {rnd} {cid} {v:24s} op {res[v]['op'][-1]:6.2f}"
                        + (f" stub {res[v]['stub'][-1]:6.2f}" if STUB else "")
                    )
            for v in VARIANTS:
                op = res[v]["op"]
                st = res[v]["stub"]
                line = (
                    f"RESULT {cid:18s} {v:24s} op med {statistics.median(op):6.2f} [{min(op):.2f}-{max(op):.2f}]"
                    + (f" stub med {statistics.median(st):6.2f} [{min(st):.2f}-{max(st):.2f}]" if st else "")
                    + f" | pcc {pccs[v][0]:.6f} exact {pccs[v][1]}"
                )
                print(line)
                summary.append(line)
    print("\n" + "\n".join(summary))
