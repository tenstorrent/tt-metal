# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment far_slice_fabric_mcast: base op (relay) vs the multicast copy, same session, eval CCL protocol.

    MC_VARIANTS=base,mcast,base,mcast CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=/tmp/x scripts/run_safe_pytest.sh \\
        --no-precompile ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/far_slice_fabric_mcast/perf_mcast.py -s

  MC_VARIANTS   comma list of base | mcast | relaycopy (the copy with AGRMS_MCAST=0, i.e. the base schedule)
  CCL_PERF_CASES focus | all | indices into the loose cases
  MC_NUM_LINKS / MC_TOPOLOGY (Ring | Linear)  axis overrides for the domain sweep
  MC_ORDER      far | near (multicast send order)
  AGRMS_ABLATE  as the op (TRANSPORT / ...): applies to every variant; no golden check then
Each un-ablated variant is checked against the golden reference before timing.

STATUS (QuietBox 2, 1x4 ring mock): the mcast variants HANG. The 1x4 mesh is a snake over the physical 2x2 torus
(fabric nodes D0 -E-> D1 -S-> D3 -W-> D2 -N-> D0, probe_fabric_directions.py), so every ring hop turns: an h-hop line
multicast in one direction lands on the wrong chips (E,E from D0 = D1, D0), and a 2-hop unicast injected on the
neighbour's link is routed for a different first hop (D0 -> D3 goes S first). evidence/: port callstacks of the hangs.
"""

from __future__ import annotations

import os

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm as base_op  # noqa: E402
from ttnn.operations.all_gather_rms_norm.perf_experiments.far_slice_fabric_mcast.op import (  # noqa: E402
    all_gather_rms_norm as copy_op,
)

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", "5"))
ABLATE = os.environ.get("AGRMS_ABLATE", "")
VARIANTS = os.environ.get("MC_VARIANTS", "base,mcast").split(",")


def _selected():
    cases = _loose_cases()
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    if sel == "all":
        return cases
    if sel == "focus":
        return cases[:1]
    return [cases[int(i)] for i in sel.split(",")]


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("inputs,axes,extras", _selected())
def test_mcast(inputs, axes, extras, mesh_cache, request):
    import ttnn

    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    mesh = mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)
    if os.environ.get("MC_NUM_LINKS"):
        axes = dict(axes, num_links=int(os.environ["MC_NUM_LINKS"]))
    if os.environ.get("MC_TOPOLOGY"):
        axes = dict(axes, topology=getattr(ttnn.Topology, os.environ["MC_TOPOLOGY"]))
    if os.environ.get("MC_ORDER"):
        os.environ["AGRMS_MCAST_ORDER"] = os.environ["MC_ORDER"]
    tag = f"{inputs[0][-2]}x{inputs[0][-1]}-L{axes['num_links']}-{str(axes['topology']).split('.')[-1]}"
    out_dir = os.environ.get("CCL_PERF_OUT")
    for i, v in enumerate(VARIANTS):
        op = base_op if v == "base" else copy_op
        os.environ["AGRMS_MCAST"] = "0" if v == "relaycopy" else "1"
        call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=op, **axes)
        out = call()
        if not ABLATE:
            check(out)
        path = None
        if out_dir:
            from pathlib import Path

            path = Path(out_dir) / f"{tag}-{ABLATE or 'full'}-{v}-{i}.json"
        ccl_perf.run(mesh, call, experiments=EXPERIMENTS, out_path=path, label=f"{tag}-{v}-{i}")
