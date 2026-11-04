# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _assert_outputs_equal(output1, output2):
    if output1 is None or output2 is None:
        assert output1 is None and output2 is None, "one run returned None, the other did not"
        return
    if isinstance(output1, (tuple, list)):
        assert len(output1) == len(output2)
        for a, b in zip(output1, output2):
            _assert_outputs_equal(a, b)
        return
    a = ttnn.to_torch(output1)
    b = ttnn.to_torch(output2)
    assert a.shape == b.shape, f"shape mismatch between the two runs: {a.shape} vs {b.shape}"
    if a.is_floating_point():
        both_nan = a.isnan() & b.isnan()
        assert bool(torch.all((a == b) | both_nan)), "the two runs produced different outputs"
    else:
        assert torch.equal(a, b), "the two runs produced different outputs"


def _run_twice(op, *args, **kwargs):
    output1 = op(*args, **kwargs)
    output2 = op(*args, **kwargs)
    _assert_outputs_equal(output1, output2)
    return output1


def ttnn_linear(*args, **kwargs):
    return _run_twice(ttnn.linear, *args, **kwargs)
