# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tt_py_test_utils_common.determinism_report import assert_deterministic


def _assert_outputs_equal(output1, output2, op=None, args=(), kwargs=None, rerun=None):
    """Assert two runs' outputs are identical (NaN at matching positions counts as equal).

    ``op``/``args``/``kwargs``/``rerun`` are optional and only enrich the failure message.
    """
    assert_deterministic(op, args, kwargs or {}, output1, output2, rerun=rerun, nan_equal=True)


def _run_twice(op, *args, **kwargs):
    output1 = op(*args, **kwargs)
    output2 = op(*args, **kwargs)
    _assert_outputs_equal(output1, output2, op, args, kwargs, lambda: op(*args, **kwargs))
    return output1


def ttnn_linear(*args, **kwargs):
    return _run_twice(ttnn.linear, *args, **kwargs)
