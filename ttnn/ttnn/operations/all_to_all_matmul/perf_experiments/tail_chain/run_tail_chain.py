# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tail_chain experiment driver (perf tournament, OVERLAP: the post-exchange chain).

Runs copies of the op (base/ = the op as it was on disk, cand/ = tail M-split) on the focus ring through the eval
harness, in one session, alternating. Not a golden test.

  VARS     ";"-separated "copy|ablate|xp" (copy = base or cand; xp = A2AMM_XP knobs, e.g. "cand||ms=4,mp=3,tail=4")
  CASES    LOOSE_CASES indices (default "0")
  EXP      harness experiments per variant (default 5)
  OUTDIR   result directory (default perf_experiments/tail_chain/out)
  CHECK    1 = check vs golden before timing (default 1)
  ZONES    1 = instead of timing, run 3 untraced calls with A2AMM_ABLATE=zones and dump the profiler log per variant

  echo "import os; os.environ['TAIL_CHAIN_RUN']='1'; exec(open('<this file>').read())" | scripts/tt-probe.sh all_to_all_matmul
"""

import importlib.util
import os
import sys
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(_k, _v)

import ttnn  # noqa: E402
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul  # noqa: E402
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases  # noqa: E402

HERE = Path(
    os.environ.get("TAIL_CHAIN_DIR", "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/tail_chain")
).resolve()


def _load(copy):
    name = f"a2amm_tail_chain_{copy}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / copy / "__init__.py", submodule_search_locations=[str(HERE / copy)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def golden_check():
    """GOLDEN="7,10,15" (feature_spec.INPUTS indices) x PAYLOADS ("7232,0": 0 = arch max, the golden default) x VARS
    (copy||xp), model config (16-bit DEST, HiFi2, L1 out, bf16) plus Phase 0 config when CFG=all: check vs golden."""
    from eval.golden_tests.all_to_all_matmul.feature_spec import INPUTS, _MODEL, _PHASE0

    cfgs = {"model": (_MODEL, ttnn.MathFidelity.HiFi2), "phase0": (_PHASE0, ttnn.MathFidelity.HiFi4)}
    sel = os.environ.get("CFG", "model")
    variants = [(v.split("|") + ["", ""])[:3] for v in os.environ["VARS"].split(";")]
    cache = _FabricMeshCache()
    try:
        for pay in [int(x) for x in os.environ.get("PAYLOADS", "7232,0").split(",")]:
            mesh = cache.get((1, 4), payload_bytes=pay or None)
            for gi in [int(i) for i in os.environ["GOLDEN"].split(",")]:
                for cname, (axes, fid) in cfgs.items():
                    if sel != "all" and cname != sel:
                        continue
                    for copy, _, xp in variants:
                        os.environ["A2AMM_XP"] = xp
                        call, check = prepare_all_to_all_matmul(
                            INPUTS[gi],
                            mesh_device=mesh,
                            extras={"math_fidelity": fid},
                            op=_load(copy).all_to_all_matmul,
                            **axes,
                        )
                        for _ in range(2):
                            check(call())
                        print("GOLDEN OK", pay, INPUTS[gi][0], INPUTS[gi][1], cname, copy, xp, flush=True)
    finally:
        os.environ["A2AMM_XP"] = ""
        cache.close()


def test_tail_chain():
    out = Path(os.environ.get("OUTDIR", str(HERE / "out")))
    out.mkdir(parents=True, exist_ok=True)
    cases = [int(c) for c in os.environ.get("CASES", "0").split(",")]
    variants = [(v.split("|") + ["", ""])[:3] for v in os.environ["VARS"].split(";")]
    zones = os.environ.get("ZONES", "0") == "1"
    cache = _FabricMeshCache()
    try:
        for ci in cases:
            inputs, axes, extras = _loose_cases()[ci].values
            mesh = cache.get(
                (1, 4),
                trace_region_size=ccl_perf.TRACE_REGION_SIZE,
                payload_bytes=extras.get("fabric_payload_bytes"),
            )
            for copy, abl, xp in variants:
                op = _load(copy).all_to_all_matmul
                call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=op, **axes)
                os.environ["A2AMM_ABLATE"] = "zones" if zones else abl
                os.environ["A2AMM_XP"] = xp
                tag = f"c{ci}_{copy}_{abl}_{xp}".replace(",", "+").replace("=", "")
                if not abl and os.environ.get("CHECK", "1") == "1":
                    check(call())
                    print("CHECK OK", tag, flush=True)
                if int(os.environ.get("B2B", "0")):
                    for i in range(int(os.environ["B2B"])):
                        o = call()
                    check(o)
                    print("B2B OK", tag, flush=True)
                    continue
                if int(os.environ.get("NCALLS", "0")):
                    for i in range(int(os.environ["NCALLS"])):
                        check(call())
                        print("CALL OK", i, tag, flush=True)
                    continue
                if zones:
                    for _ in range(3):
                        call()
                    ttnn.synchronize_device(mesh)
                    ttnn.ReadDeviceProfiler(mesh)
                    # the CSV (generated/profiler/.logs/profile_log_device.csv) is written when the mesh closes:
                    # run one variant per process and copy it after the run
                    continue
                ccl_perf.run(
                    mesh, call, experiments=int(os.environ.get("EXP", "5")), out_path=out / f"{tag}.json", label=tag
                )
    finally:
        os.environ["A2AMM_ABLATE"] = ""
        os.environ["A2AMM_XP"] = ""
        cache.close()


if __name__ == "__main__" or os.environ.get("TAIL_CHAIN_RUN") == "1":
    golden_check() if os.environ.get("GOLDEN") else test_tail_chain()
