# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""graduation.patch check: op_grad/ (the real op at 2ac42333eb2 + graduation.patch) against op_copy/'s defaults (= the
committed op), same cases and run log as test_tdg.py. Variants: `baseline` (op_copy, default knobs), `grad` (op_grad).
TDG_CASES / TDG_VARIANTS / TDG_ORDER / TDG_LOG as in test_tdg.py. Run with --import-mode=prepend."""

import importlib

import pytest

T = importlib.import_module("ttnn.operations.conv2d_3x3.perf_experiments.tap_direct_gather_hide.test_tdg")
_GRAD = "ttnn.operations.conv2d_3x3.perf_experiments.tap_direct_gather_hide.op_grad"
_grad_op = importlib.import_module(_GRAD + ".conv2d_3x3")
_grad_pd = importlib.import_module(_GRAD + ".conv2d_3x3_program_descriptor")
_copy_op, _copy_pd = T.op, T.pd

PARAMS = [(c, v) for c, v in T.PARAMS if v in ("baseline", "grad")]


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_grad(case_id, variant, device, monkeypatch):
    grad = variant == "grad"
    monkeypatch.setattr(T, "op", _grad_op if grad else _copy_op)
    monkeypatch.setattr(T, "pd", _grad_pd if grad else _copy_pd)
    if not grad:
        T.apply_variant("baseline", monkeypatch)
    _grad_pd.select_blocking.cache_clear()
    _run(case_id, variant, device, monkeypatch)


def _run(case_id, variant, device, monkeypatch):
    # test_tdg's body with VARIANTS lookups skipped for `grad`.
    import hashlib
    import json

    import torch
    import ttnn
    from tests.ttnn.utils_for_testing import comp_pcc

    info = dict(case=case_id, variant=variant, ablate="")
    T._record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold = T._run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold = T._run_focus(T.FOCUS[case_id], device)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    T.RUN_LOG.parent.mkdir(exist_ok=True)
    with open(T.RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info
