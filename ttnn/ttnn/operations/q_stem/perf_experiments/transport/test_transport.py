# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""q_stem transport bake-off on the focus ring (1x4, FABRIC_2D_TORUS_XY, 7232 B router payload, 2 links).

    TP_VARIANTS=baseline,balanced TP_EXPERIMENTS=3 scripts/run_safe_pytest.sh --no-precompile \
        ttnn/ttnn/operations/q_stem/perf_experiments/transport/test_transport.py -s

TP_CASE kimi (focus) | dsv32 | glm; TP_PAYLOAD (default 7232); TP_OUT result dir (ccl_perf JSON per variant).
Every variant's landing is checked word-exactly on every chip before it is timed (eval/ccl_perf protocol).
"""

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.q_stem.conftest import _FabricMeshCache  # noqa: E402
from eval.golden_tests.q_stem.topology import focus_mesh_shape  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("q_stem_transport_bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

VARIANTS = [v for v in os.environ.get("TP_VARIANTS", "baseline,balanced").split(",") if v]
CASE = os.environ.get("TP_CASE", "kimi")
PAYLOAD = int(os.environ.get("TP_PAYLOAD", "7232"))
EXPERIMENTS = int(os.environ.get("TP_EXPERIMENTS", "3"))
L = int(os.environ.get("TP_LINKS", "2"))
PLACEMENT = os.environ.get("TP_PLACEMENT", "centre")
OUT = Path(os.environ.get("TP_OUT", "ttnn/ttnn/operations/q_stem/perf_experiments/transport/results"))


@pytest.fixture(scope="module")
def mesh():
    cache = _FabricMeshCache()
    shape, why = focus_mesh_shape()
    if shape is None:
        pytest.skip(why)
    m = cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=PAYLOAD)
    yield m
    cache.close()


def test_transport(mesh):
    if os.environ.get("TP_PROBE"):
        import ttnn
        from ttnn.operations.q_stem import q_stem_program_descriptor as qpd

        for s in range(4):
            for t in range(4):
                if s != t:
                    n_s, n_t = qpd._node(mesh, (0, s)), qpd._node(mesh, (0, t))
                    print(
                        f"DIR {s}->{t}: chip {n_s.chip_id}->{n_t.chip_id} dir {ttnn.get_eth_forwarding_direction(n_s, n_t)}"
                        f" links {ttnn.get_forwarding_link_indices(n_s, n_t)}",
                        flush=True,
                    )
        return
    geo = bench.geometry(mesh, CASE)
    res = bench.Resources(mesh, geo, L, PLACEMENT)
    ports = {p: {k: (c.x, c.y) for k, c in v.items()} for p, v in res.ports_of.items()}
    print(
        f"\n=== q_stem transport  case={CASE} G={geo.G} Hl={geo.Hl} hw={geo.hw} rows={geo.rows_per_head} "
        f"payload={PAYLOAD} links={L} grid={geo.gx}x{geo.gy} hpc_max={geo.hpc_max} placement={PLACEMENT} ports {ports}",
        flush=True,
    )
    for variant in VARIANTS:
        _, sends, coords, chips = bench.create(res, variant, PAYLOAD)
        load = bench.link_load(res, sends, coords, chips)
        busiest = max(load.values()) if load else 0
        sent = {
            p: sum(
                len(h) * geo.rows_per_head * res.seg_bytes
                for (pp, _, _), es in sends.items()
                if pp == p
                for _, _, h in es
            )
            for p in range(geo.G)
        }
        nbytes = bench.check(res, variant, PAYLOAD)
        loads = {f"{a}{'+' if d == 0 else '-'}L{l}": round(b / 1e6, 2) for (a, d, l), b in sorted(load.items())}
        print(
            f"TRANSPORT {variant}: landing exact ({nbytes} B received over the ring); bytes sent per chip "
            f"{[round(b / 1e6, 3) for b in sent.values()]} MB; busiest directed link {busiest / 1e6:.3f} MB; "
            f"link loads MB {loads}",
            flush=True,
        )
        call = lambda: bench.run_once(res, variant, PAYLOAD)  # noqa: E731
        doc = ccl_perf.run(
            mesh,
            call,
            experiments=EXPERIMENTS,
            out_path=OUT / CASE / f"{variant}__{PLACEMENT}__L{L}.json",
            label=f"{CASE}:{variant}",
            meta={"variant": bench.VARIANTS[variant], "case": CASE, "busiest_link_bytes": busiest, "sent": sent},
        )
        s = doc["summary"]
        fast, slow = s["fastest"]["mean"], s["slowest"]["mean"]
        per = {k: round(v["mean"], 2) for k, v in s["per_chip"].items()}
        print(
            f"TRANSPORT_RESULT {CASE} {PLACEMENT} L{L} {variant:16s} fastest {fast:8.2f} us  slowest {slow:8.2f} us  per_chip {per}  "
            f"busiest-link {busiest / slow / 1e3:6.2f} GB/s (at slowest) {busiest / fast / 1e3:6.2f} (at fastest)  "
            f"per-chip all-to-all {sent[0] / slow / 1e3:6.2f} GB/s",
            flush=True,
        )
