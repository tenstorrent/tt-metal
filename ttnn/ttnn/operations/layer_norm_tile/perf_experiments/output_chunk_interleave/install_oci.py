# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Route layer_norm_tile through pd_oci (kernels in ./op_kernels) unless LN_OCI=real; log precision per call."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import eval.golden_tests.layer_norm_tile.helpers as _helpers  # noqa: E402

PATCHED = os.environ.get("LN_OCI", "real") != "real"
NAME = os.environ.get("LN_OCI", "real")

if PATCHED:
    import pd_oci  # noqa: E402

    _orig_create = pd_oci.create_program_descriptor
    _calls = [0]

    def _create(*args, **kwargs):
        _calls[0] += 1
        if _calls[0] in (1, 100, 300):
            print(f"\nOCI_ACTIVE variant={NAME} calls={_calls[0]}")
        return _orig_create(*args, **kwargs)

    pd_oci.create_program_descriptor = _create

    _orig_layer_norm = _helpers.layer_norm

    def _patched_layer_norm(*args, **kwargs):
        # The op module is imported under two names (import-mode=prepend): patch the one the golden
        # helpers' observe wrapper dispatches into.
        import eval.golden_tests.layer_norm_tile.axes as axes

        op_globals = axes._raw.__globals__
        op_globals["create_program_descriptor"] = pd_oci.create_program_descriptor
        op_globals["select_blocking"] = pd_oci.select_blocking
        return _orig_layer_norm(*args, **kwargs)

    _helpers.layer_norm = _patched_layer_norm

    import eval.golden_tests.layer_norm_tile.axes as _axes  # noqa: E402

    _axes._raw.__globals__["create_program_descriptor"] = pd_oci.create_program_descriptor
    _axes._raw.__globals__["select_blocking"] = pd_oci.select_blocking

_orig_check = _helpers.check_output


def _check_and_log(*args, **kwargs):
    metrics = _orig_check(*args, **kwargs)
    if metrics is not None:
        print(f"\nOCI_PREC variant={NAME} pcc={metrics.pcc:.7f} rel_rms={metrics.rms:.6f}")
    return metrics


_helpers.check_output = _check_and_log
