# Harness (eval/ccl_perf.py: spin + traced calls) for each variant in one session; golden check before timing.
#   VARS="base;sscopy;...;base"  EXP=5  OUTDIR=/tmp/ss_perf
import os
from pathlib import Path

from eval import ccl_perf

for k, v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(k, v)
import importlib.util  # noqa: E402

from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul  # noqa: E402
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases  # noqa: E402
from ttnn.operations.all_to_all_matmul import all_to_all_matmul  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ss_common", Path(__file__).parent / "_common.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def test_perf():
    inputs, axes, extras = _loose_cases()[int(os.environ.get("CASE", "0"))].values
    cache = _FabricMeshCache()
    mesh = cache.get(
        (1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes")
    )
    call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
    out = Path(os.environ.get("OUTDIR", "/tmp/ss_perf"))
    seen = {}
    for v in os.environ.get("VARS", "base").split(";"):
        toks = [t for t in v.split(",") if t and t != "base"]
        os.environ["A2AMM_ABLATE"] = ",".join(toks)
        if all(t.startswith("ss") for t in toks):  # payload stubs (nowrite, ...) make the output wrong by design
            check(call())
            print("CHECK OK", v, flush=True)
        seen[v] = seen.get(v, 0) + 1
        ccl_perf.run(
            mesh,
            call,
            experiments=int(os.environ.get("EXP", "5")),
            out_path=out / f"{v.replace(',', '+')}_{seen[v]}.json",
            label=v,
        )
    os.environ["A2AMM_ABLATE"] = ""
    cache.close()


if __name__ == "__main__" or True:
    test_perf()
