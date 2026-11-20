# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The FLUX.2 comparison harness (tests/.../test_flux_conv3x3_compare.py) on this experiment's op copy.

  BRF_BANK_RUN=1 BRF_TEST=<this file> run_flux.sh <label> 41,42,35,47 --import-mode=prepend
BRF_BANK_RUN=0 is the copy's baseline (the committed per-pixel fill). Never profile next to the real op."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import brf_load  # noqa: E402

opc, _pd, _op = brf_load.load()

REPO = Path(__file__).resolve().parents[6]
_spec = importlib.util.spec_from_file_location(
    "brf_flux_cmp", REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.conv2d_3x3 = opc.conv2d_3x3
_mod.prepare_conv2d_weights = opc.prepare_conv2d_weights
_mod.prepare_conv2d_bias = opc.prepare_conv2d_bias

test_flux_conv3x3_compare = _mod.test_flux_conv3x3_compare

# BRF_HASH=1: record a hash of every conv2d_3x3 output (whole tensor, raw bf16 bytes) per case and BRF_BANK_RUN, for
# the bit-identity check between the copy's baseline and the bank-run fill (hash_check.py).
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

_orig_pcc_bands = _mod._pcc_bands


def _hashing_pcc_bands(out_flat, *args, **kwargs):
    if os.environ.get("BRF_HASH") == "1":
        h = hashlib.sha1(out_flat.contiguous().view(-1).view(dtype=__import__("torch").int16).numpy().tobytes())
        rec = dict(
            shape=list(out_flat.shape),
            args=[int(a) if isinstance(a, int) else None for a in args[3:]],
            bank_run=os.environ.get("BRF_BANK_RUN", "1"),
            sha1=h.hexdigest(),
        )
        with open(Path(__file__).parent / "runs" / "hashes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return _orig_pcc_bands(out_flat, *args, **kwargs)


_mod._pcc_bands = _hashing_pcc_bands


# BRF_PRINT_BLOCKING=1: log each solved blocking (bm, bk, footprint) to runs/blocking.jsonl.
_orig_sb = _pd.select_blocking


def _logging_select_blocking(g, grid, l1_budget):
    b = _orig_sb(g, grid, l1_budget)
    if os.environ.get("BRF_PRINT_BLOCKING") == "1":
        with open(Path(__file__).parent / "runs" / "blocking.jsonl", "a") as f:
            f.write(
                json.dumps(dict(g=repr(g)[:200], bank_run=os.environ.get("BRF_BANK_RUN", "1"), b=repr(b)[:600])) + "\n"
            )
    return b


_pd.select_blocking = _logging_select_blocking
if hasattr(_op, "select_blocking"):
    _op.select_blocking = _logging_select_blocking
