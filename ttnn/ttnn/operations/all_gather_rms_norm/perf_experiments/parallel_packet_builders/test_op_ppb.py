# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole op with parallel packet builders on the own entry, by the eval harness (perf experiment only).

Runs eval/golden_tests/all_gather_rms_norm/perf_measure.py's test with the op's program descriptor swapped for
op_copy_descriptor.py (this dir). PPB_H=0 there is the op unchanged (same kernels, same harness session):

    PPB_H=4 CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile \
        ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/parallel_packet_builders/test_op_ppb.py -s
"""

import ttnn

# ttnn.operations auto-imports every module under it during ttnn's own init: act only once ttnn is complete
if hasattr(ttnn, "Topology"):
    import importlib

    _op_mod = importlib.import_module("ttnn.operations.all_gather_rms_norm.all_gather_rms_norm")

    from ttnn.operations.all_gather_rms_norm.perf_experiments.parallel_packet_builders import (
        op_copy_descriptor as _copy,
    )

    _op_mod.create_mesh_program_descriptor = _copy.create_mesh_program_descriptor
    assert (
        _op_mod.all_gather_rms_norm.__globals__["create_mesh_program_descriptor"]
        is _copy.create_mesh_program_descriptor
    )

    from eval.golden_tests.all_gather_rms_norm.perf_measure import *  # noqa: E402,F401,F403
