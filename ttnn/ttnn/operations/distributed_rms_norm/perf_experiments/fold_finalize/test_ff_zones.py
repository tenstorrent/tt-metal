# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_experiments/fold_finalize: the fold hop on the critical owner, from DRMS_PERF_ZONES=1 zones (op copy).

Per variant (FF_VARIANTS, ';'-separated; "base" = the real op): FF_ZONE_CALLS untraced calls; for each call, on the
slowest chip, the owner fold whose all-gather starts last (the critical one):
  collect end (NCRISC rd_collect) -> TRISC_0 cmp_fold start (records visible) -> TRISC_0 / TRISC_1 / TRISC_2
  cmp_fold end -> NCRISC rd_allgather start; plus the medians over every owned fold of that chip.

    FF_VARIANTS="copy;compact" scripts/run_safe_pytest.sh --no-precompile --run-all \\
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/fold_finalize/test_ff_zones.py -s --import-mode=prepend
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import csv
import importlib.util as _ilu
import pathlib as _pl
import statistics
from collections import defaultdict

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as op_base

import ttnn.operations.distributed_rms_norm.perf_experiments.fold_finalize.drms_ff as cp

_UT = _pl.Path(__file__).resolve().parents[6] / "tests/ttnn/unit_tests/operations/distributed_rms_norm"
_spec = _ilu.spec_from_file_location("_drms_accept", _UT / "test_distributed_rms_norm.py")
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

LOG = "generated/profiler/.logs/profile_log_device.csv"
VARIANTS = [v for v in os.environ.get("FF_VARIANTS", "copy;compact").split(";") if v]
CALLS = int(os.environ.get("FF_ZONE_CALLS", "4"))
M, W = [int(v) for v in os.environ.get("FF_SHAPE", "640,1792").split(",")]


def _analyze():
    with open(LOG) as fh:
        mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    by = defaultdict(list)
    for r in rows:
        by[(int(r["PCIe slot"]), int(r["run host ID"]))].append(r)
    out = {}
    for chip in sorted({c for c, _ in by}):
        rid = max(rr for c, rr in by if c == chip)
        stack, inst = defaultdict(list), defaultdict(list)
        for r in by[(chip, rid)]:
            key = (int(r["core_x"]), int(r["core_y"]), r["RISC processor type"], r["zone name"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                stack[key].append(t)
            elif r["type"] == "ZONE_END" and stack[key]:
                inst[key].append((stack[key].pop(), t))
        t0 = min(s for k, v in inst.items() if k[3].endswith("-KERNEL") for s, _ in v)
        kend = max(e for k, v in inst.items() if k[3].endswith("-KERNEL") for _, e in v)
        ns = lambda c: (c - t0) * 1000.0 / mhz
        folds = []
        cores = {(x, y) for (x, y, rk, z) in inst if z == "cmp_fold"}
        for x, y in cores:
            f0 = sorted(inst[(x, y, "TRISC_0", "cmp_fold")])
            f1 = sorted(inst[(x, y, "TRISC_1", "cmp_fold")])
            f2 = sorted(inst[(x, y, "TRISC_2", "cmp_fold")])
            ag = sorted(inst[(x, y, "NCRISC", "rd_allgather")])
            col = sorted(inst[(x, y, "NCRISC", "rd_collect")])
            for i in range(min(len(f0), len(f1), len(f2), len(ag), len(col))):
                folds.append(
                    dict(
                        core=(x, y),
                        i=i,
                        col_end=ns(col[i][1]),
                        u0=ns(f0[i][0]),
                        u1=ns(f0[i][1]),
                        m1=ns(f1[i][1]),
                        p1=ns(f2[i][1]),
                        ag=ns(ag[i][0]),
                    )
                )
        out[chip] = (ns(kend), folds)
    return out


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
def test_ff_zones(mesh_device):
    import torch

    mesh = mesh_device
    os.environ["DRMS_PERF_ZONES"] = "1"
    shape = (1, 1, M, W)
    x = _acc._stacked((1, 4), shape, torch.bfloat16)
    tx = _acc._to_mesh(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gamma = _acc._stacked((1, 4), (1, 1, 1, W), torch.bfloat16)
    tg = _acc._gamma_to_mesh(gamma, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    lines = []
    for rnd in range(2):
        for v in VARIANTS if rnd == 0 else VARIANTS[::-1]:
            fn = op_base if v == "base" else cp.distributed_rms_norm
            os.environ["DRMS_FF_VARIANT"] = "base" if v in ("base", "copy") else v
            hop, fa, crit = [], [], []
            for c in range(CALLS + 1):
                fn(tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2)
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                if c == 0:
                    continue  # compile
                res = _analyze()
                slow = max(res, key=lambda ch: res[ch][0])
                kend, folds = res[slow]
                f = max(folds, key=lambda d: d["ag"])
                crit.append(f)
                hop += [d["p1"] - d["u0"] for d in folds]
                fa += [d["ag"] - d["u0"] for d in folds]
                print(
                    f"ZCALL {v:10s} chip {slow} kend {kend/1000:6.2f}us crit core {f['core']} w{f['i']}: collect_end "
                    f"{f['col_end']:7.0f} fold u0 {f['u0']:7.0f}->{f['u1']:7.0f} math_end {f['m1']:7.0f} pack_end "
                    f"{f['p1']:7.0f} allgather_start {f['ag']:7.0f} | fold hop {f['p1']-f['u0']:5.0f} "
                    f"fold->ag {f['ag']-f['u0']:5.0f} collect_end->ag {f['ag']-f['col_end']:5.0f} ns"
                )
            med = lambda k: statistics.median(k)
            line = (
                f"ZRESULT {v:10s} crit: fold hop {med([d['p1']-d['u0'] for d in crit]):5.0f} "
                f"fold->ag {med([d['ag']-d['u0'] for d in crit]):5.0f} "
                f"collect_end->ag {med([d['ag']-d['col_end'] for d in crit]):5.0f} TRISC_0 fold "
                f"{med([d['u1']-d['u0'] for d in crit]):5.0f} | all owned folds: hop {med(hop):5.0f} fold->ag {med(fa):5.0f}"
            )
            print(line)
            lines.append(line)
    os.environ.pop("DRMS_FF_VARIANT", None)
    os.environ.pop("DRMS_PERF_ZONES", None)
    print("\n" + "\n".join(lines))
