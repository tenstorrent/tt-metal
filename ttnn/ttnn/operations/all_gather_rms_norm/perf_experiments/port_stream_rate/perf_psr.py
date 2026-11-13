# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment (port_stream_rate): the eval CCL harness protocol (eval/ccl_perf.py) on a COPY of
all_gather_rms_norm (agrms_copy/), with the pre-send variants selected by env (see agrms_copy's descriptor):

    AGRMS_PRESEND_A=1            route one packet header, copy it to the others (lever a)
    AGRMS_PRESEND_DRAM_ITEMS=N   the port reader streams the first N own items straight from DRAM (lever b)
    AGRMS_PRESEND_SPLIT_FENCE=1  send the ready before routing / opening anything else (lever c)

    CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile <this file> -s

Overrides for the domain sweep: CCL_PERF_CASES (focus | all | indices), AGRMS_PERF_SHAPE, AGRMS_PERF_NUM_LINKS,
AGRMS_PERF_MEMORY_LAYOUT, AGRMS_PERF_TOPOLOGY (Ring | Linear). Every case is checked against the golden first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from agrms_copy import all_gather_rms_norm  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", "5"))
TAG = os.environ.get("PSR_TAG", "x")


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
def test_psr(inputs, axes, extras, mesh_cache, request):
    import ttnn

    if os.environ.get("AGRMS_PERF_TOPOLOGY"):
        axes = dict(axes, topology=getattr(ttnn.Topology, os.environ["AGRMS_PERF_TOPOLOGY"]))
    if os.environ.get("AGRMS_PERF_NUM_LINKS"):
        axes = dict(axes, num_links=int(os.environ["AGRMS_PERF_NUM_LINKS"]))
    if os.environ.get("AGRMS_PERF_MEMORY_LAYOUT"):
        axes = dict(axes, memory_layout=getattr(ttnn.TensorMemoryLayout, os.environ["AGRMS_PERF_MEMORY_LAYOUT"]))
    if os.environ.get("AGRMS_PERF_SHAPE"):
        inputs = (tuple(int(v) for v in os.environ["AGRMS_PERF_SHAPE"].split(",")),)
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, why = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        if resolved is None:
            pytest.skip(why)
        shape = resolved
    mesh = mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)
    call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes)
    ablate = os.environ.get("AGRMS_ABLATE", "")
    if ablate:
        check = lambda out: None
    check(call())
    check(call())  # second call: the ready fence between calls, and a program-cache hit
    out_dir = os.environ.get("CCL_PERF_OUT")
    path = None
    if out_dir:
        ml = os.environ.get("AGRMS_PERF_MEMORY_LAYOUT", "")
        name = f"{inputs[0][-2]}x{inputs[0][-1]}-{axes['topology']}-L{axes['num_links']}{ml and '-' + ml}{ablate and '-' + ablate}-{TAG}.json"
        path = Path(out_dir) / name.replace("Topology.", "")
    ccl_perf.run(mesh, call, experiments=EXPERIMENTS, out_path=path, label=f"{request.node.callspec.id}-{TAG}")
    # correctness after the timed calls too (every chip's output of the last traced call)
    check(call())
