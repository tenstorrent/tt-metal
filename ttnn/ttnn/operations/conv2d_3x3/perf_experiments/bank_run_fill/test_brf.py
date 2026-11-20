# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""bank_run_fill bench: correctness (bit-exact slab vs input) and device time of each fill / de-interleave variant.

  BRF_CASES=41 scripts/run_safe_pytest.sh --run-all <this file> -k verify
  BRF_CASES=41 BRF_VARIANTS=base,bank scripts/run_safe_pytest.sh --profile --run-all <this file> -k perf
then python3 report_brf.py <ops_perf_results csv>.
Env: BRF_CHUNK_BYTES (staging chunk, default 64 KB), BRF_NSTG (staging chunks, default 4), BRF_VERIFY_CORES.
"""

import os
import sys
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

sys.path.insert(0, str(Path(__file__).parent))
import bench_brf as B  # noqa: E402

CASE_IDS = [int(c) for c in os.environ.get("BRF_CASES", "41").split(",")]
PERF_VARIANTS = os.environ.get(
    "BRF_VARIANTS", "base,bank,bank_noc,bank_risc,bank_trisc,noc_only,risc_only,trisc_only,trisc_tilize_only"
).split(",")
VERIFY_VARIANTS = os.environ.get("BRF_VERIFY_VARIANTS", "base,bank_noc,bank_risc,bank_trisc").split(",")
CHUNK_BYTES = int(os.environ.get("BRF_CHUNK_BYTES", "65536"))
NSTG = int(os.environ.get("BRF_NSTG", "4"))
VERIFY_CORES = int(os.environ.get("BRF_VERIFY_CORES", "110"))


def _collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _input(device, g, zeros=False):
    import torch

    n = g["num_in_px"]
    x = torch.zeros(n, g["C"], dtype=torch.bfloat16) if zeros else torch.randn(n, g["C"], dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x.reshape(1, 1, n, g["C"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x, t


def _out(device, g, pages, poison=False):
    if poison:
        # A known non-input pattern, so a variant cannot pass on another variant's stale output.
        return ttnn.full(
            [1, 1, pages, g["band_cap"] * g["C"]],
            fill_value=-777.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, pages, g["band_cap"] * g["C"]]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def _kw(variant, g):
    ch, nstg = B.chunk_params(g["pb"], CHUNK_BYTES, NSTG)
    mode = 1 if variant.endswith("tilize_only") else 0
    return dict(ch=ch, nstg=nstg, mode=mode)


@pytest.mark.parametrize("idx", CASE_IDS)
def test_brf_verify(device, idx):
    import torch

    g = B.geometry(idx)
    collect = _collect_pass()
    torch.manual_seed(0)
    x, tin = _input(device, g, zeros=collect)
    pages = 110 * g["max_blocks"] * 3
    for variant in VERIFY_VARIANTS:
        tout = _out(device, g, pages, poison=True)
        kw = _kw(variant, g)
        B.run(g, variant, tin, tout, verify=True, verify_cores=VERIFY_CORES, **kw)
        if collect:
            continue
        got = ttnn.to_torch(tout).reshape(pages, g["band_cap"], g["C"])
        bad = 0
        checked = 0
        for c in range(min(VERIFY_CORES, 110)):
            start, actual = g["cores"][c]
            nb = (actual + g["bm"] - 1) // g["bm"]
            for m in range(nb):
                for kh, (lo, hi) in enumerate(B.bands(g, start, actual, m)):
                    page = (c * g["max_blocks"] + m) * 3 + kh
                    checked += 1
                    if not torch.equal(got[page, : hi - lo], x[lo:hi]):
                        bad += 1
                        if bad <= 3:
                            diff = (got[page, : hi - lo] != x[lo:hi]).any(dim=1).nonzero().flatten()
                            print(f"MISMATCH {variant} core {c} m {m} kh {kh} [{lo},{hi}) rows {diff[:8].tolist()}")
        print(f"VERIFY idx{idx} {variant} ch={kw['ch']} nstg={kw['nstg']}: {checked - bad}/{checked} bands bit-exact")
        assert bad == 0, f"{variant}: {bad} bands differ"
        ttnn.deallocate(tout)


@pytest.mark.parametrize("idx", CASE_IDS)
def test_brf_perf(device, idx):
    import torch

    g = B.geometry(idx)
    collect = _collect_pass()
    torch.manual_seed(0)
    _, tin = _input(device, g, zeros=collect)
    tout = _out(device, g, 1)
    for variant in PERF_VARIANTS:
        kw = _kw(variant, g)
        signpost(header=f"brf|idx{idx}|{variant}|ch{kw['ch']}|n{kw['nstg']}")
        B.run(g, variant, tin, tout, verify=False, **kw)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    signpost(header="brfend")
