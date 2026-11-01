# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routes layer_norm_tile through the private pd3p copy (kernels in ./kernels) and logs precision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pd3p  # noqa: E402
import eval.golden_tests.layer_norm_tile.helpers as _helpers  # noqa: E402

_orig_layer_norm = _helpers.layer_norm


def _patched_layer_norm(*args, **kwargs):
    # The op module can be imported under two names (import-mode=prepend); patch the one the
    # golden helpers' observe wrapper actually dispatches into, at call time.
    import eval.golden_tests.layer_norm_tile.axes as axes

    op_globals = axes._raw.__globals__
    op_globals["create_program_descriptor"] = pd3p.create_program_descriptor
    op_globals["select_blocking"] = pd3p.select_blocking
    return _orig_layer_norm(*args, **kwargs)


_helpers.layer_norm = _patched_layer_norm

_orig_check = _helpers.check_output


def _check_and_log(*args, **kwargs):
    try:
        metrics = _orig_check(*args, **kwargs)
        print(f"\nLN3P mode={pd3p.MODE} ablate={pd3p.ABLATE} pcc={metrics.pcc:.6f} rel_rms={metrics.rms:.5f}")
        return metrics
    except Exception as err:
        m = getattr(err, "metrics", None)
        if m is not None:
            print(f"\nLN3P mode={pd3p.MODE} FAIL pcc={m.pcc} rel_rms={m.rms}")
        if pd3p.ABLATE:
            return None  # ablated runs are timing-only
        raise


_helpers.check_output = _check_and_log
