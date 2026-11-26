# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""launch_gap: per-call wall time of structural variants under back-to-back trace replay.

LG_VARIANTS = comma list of variant names (see VARIANTS); LG_K = calls captured per trace (default 1).
Per call = wall / (replays * K). With K = 1 this is the op's trace-timing metric (82.1 us baseline)."""
import os
import time

import pytest
import ttnn

from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap import lg_descriptor as D
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap.lg_op import lg_op
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs

BASE = dict(D.KN)
BASE_NOC = D.READ_NOC_SPLIT
BASE_MUX = D.MUX_OVERFLOW_HEADS
S = dict(stub=True)
VARIANTS = {
    "real": {},
    "real_nocnone": dict(_noc="none"),
    "real_semmerge": dict(sem="merge"),
    "real_ktable": dict(worker_rt="ktable"),
    "real_ktable_sem": dict(worker_rt="ktable", sem="merge"),
    "real_mux6": dict(_mux=6),
    "real_mux6_ktable_sem": dict(_mux=6, worker_rt="ktable", sem="merge"),
    "real_ktable_sem_coloc": dict(worker_rt="ktable", sem="merge", coloc=True),
    "real_mux6c_ktable_sem": dict(_mux=6, worker_rt="ktable", sem="merge", mux_contig=True),
    "stub": S,
    "stub_nocnone": dict(S, _noc="none"),
    "stub_common": dict(S, worker_rt="common"),
    "stub_nort": dict(S, worker_rt="none"),
    "stub_nofab": dict(S, tx="no_fabric_args"),
    "stub_notx": dict(S, tx="none"),
    "stub_onecb": dict(S, cbs="one"),
    "stub_nocb": dict(S, cbs="none"),
    "stub_nocompute": dict(S, compute=False),
    "stub_min": dict(S, _noc="none", worker_rt="none", tx="none", cbs="none", compute=False),
    "stub_min_c": dict(S, _noc="none", worker_rt="none", tx="none", cbs="none"),
    "stub_8w": dict(S, workers=8),
    "trivial": dict(trivial=True),
    "stub_table": dict(S, worker_rt="table"),
    "stub_semmerge": dict(S, sem="merge"),
    "stub_nofab_sem": dict(S, tx="no_fabric_args", sem="add2"),
    "stub_fab_nosem": dict(S, sem="drop"),
    "stub_table_sem": dict(S, worker_rt="table", sem="merge"),
    "stub_table_sem_txr": dict(S, worker_rt="table", sem="merge", txr_rt="common"),
}
NAMES = [v for v in os.environ.get("LG_VARIANTS", "real,stub,trivial").split(",") if v]
K = int(os.environ.get("LG_K", "1"))
N_REPLAYS = int(os.environ.get("LG_N", "50"))


def apply(name):
    D.KN.clear()
    D.KN.update(BASE)
    cfg = dict(VARIANTS[name])
    D.READ_NOC_SPLIT = cfg.pop("_noc", BASE_NOC)
    D.MUX_OVERFLOW_HEADS = cfg.pop("_mux", BASE_MUX)
    D.KN.update(cfg)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 22}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_lg_timing(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    res = []
    for name in NAMES:
        apply(name)
        mesh_device.clear_program_cache()
        ttnn.deallocate(lg_op(attn, w))
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [lg_op(attn, w) for _ in range(K)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(N_REPLAYS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / (N_REPLAYS * K) * 1e6
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        res.append((name, us))
        print(f"\nLG_TIMING {name} K={K}: {us:.2f} us per call", flush=True)
    apply("real")
    print("\nLG_SUMMARY K=%d " % K + " | ".join(f"{n}={u:.2f}" for n, u in res))
