# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness of op_grad/ (the graduation candidate) against op_base/ (HEAD 4f2f03f6d20) on every one of the 22
[perf] focus cases, in one process, no profiler: the focus test body (PCC asserted) runs op_base, then op_grad, and the
two outputs must match element for element. Appends exact_grad.jsonl.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_startup/test_kws_exact.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kws_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
OTHER = os.environ.get("KWS_EXACT_OTHER", "grad")


def _run(case, device, monkeypatch, pkg):
    import ttnn

    device.clear_program_cache()
    op_mod, pd_mod = copy_modules(pkg)
    for name in dir(pd_mod):
        m = getattr(pd_mod, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
    for name in ("prepare_conv2d_bias", "prepare_conv2d_weights"):
        monkeypatch.setattr(perf, name, getattr(op_mod, name))
    captured = {}
    real = op_mod.conv2d_3x3

    def capture(*a, **k):
        out = real(*a, **k)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(perf, "conv2d_3x3", capture)
    perf.test_conv2d_3x3_perf_focus(case, device)
    return captured["out"]


@pytest.mark.parametrize("case", perf._FOCUS_CASES, ids=perf._FOCUS_IDS)
def test_kws_exact(case, device, monkeypatch):
    base = _run(case, device, monkeypatch, "base")
    other = _run(case, device, monkeypatch, OTHER)
    diff = int((base != other).sum().item()) if base.shape == other.shape else -1
    with open(HERE / "exact_grad.jsonl", "a") as f:
        f.write(json.dumps(dict(case=case["extras"]["layer"], other=OTHER, differ=diff, numel=base.numel())) + "\n")
    assert diff == 0, f"{case['extras']['layer']}: {OTHER} differs from base in {diff} elements"
