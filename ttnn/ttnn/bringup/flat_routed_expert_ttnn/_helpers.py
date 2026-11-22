# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small helpers the flat expert's Python side used from the MiMo-V2 perf tests (mstaletovic/mimo-v2-dp:
models/demos/mimo_v2_d_p/tests/perf/test_dram_read_fwd.py, test_stream_matmul.py), kept here so the op does not
depend on a model folder."""

import ttnn

NOC_X, NOC_Y = 17, 12  # Blackhole NoC torus (translated worker coords live inside it)
BF8_TILE = 1088  # bytes of a bfloat8_b tile


def noc_hops(src, dst, noc):
    """Hops from src to dst along one NoC torus: NOC0 travels +x then +y, NOC1 -x then -y (both wrap)."""
    if noc == 0:
        return (dst.x - src.x) % NOC_X + (dst.y - src.y) % NOC_Y
    return (src.x - dst.x) % NOC_X + (src.y - dst.y) % NOC_Y


def crs_single(cores):
    """One CoreRange per core."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
