# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at this experiment's op copy. The variant is picked by BWR_* environment variables read by the copy's
descriptor (op_copy/conv2d_3x3_program_descriptor.py, "block_weight_read_rate").

Its PCC check is wrapped to also log (test id, variant, PCC, sha256 of the output) to runs/$BWR_LOG (default
precision.jsonl), so variants can be checked bit-identical against the baseline."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bwr_copy import retarget  # noqa: E402

_MOD = "tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf"
SWAPPED = retarget(_MOD, globals())
assert "conv2d_3x3" in SWAPPED, SWAPPED

_m = importlib.import_module(_MOD)
_orig_assert = _m.assert_with_pcc


def _logged_assert(ref, out, pcc):
    from tests.ttnn.utils_for_testing import comp_pcc

    ok, value = comp_pcc(ref, out, pcc)
    rec = dict(
        test=os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
        variant=os.environ.get("BWR_VARIANT", "baseline"),
        pcc=float(value) if not isinstance(value, str) else value,
        sha=hashlib.sha256(out.contiguous().float().numpy().tobytes()).hexdigest()[:16],
        mae=float((ref.float() - out.float()).abs().mean()),
    )
    log = Path(__file__).resolve().parent / "runs" / os.environ.get("BWR_LOG", "precision.jsonl")
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return _orig_assert(ref, out, pcc)


_m.assert_with_pcc = _logged_assert
