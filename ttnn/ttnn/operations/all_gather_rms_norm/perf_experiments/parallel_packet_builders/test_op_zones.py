# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone profile of the op copy (op_copy_descriptor.py, PPB_H helpers per port; 0 = the op unchanged).

    AGRN_PROFILE_ZONES=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> PPB_H=4 \
        scripts/run_safe_pytest.sh --no-precompile <this file> -s
then tests/ttnn/unit_tests/operations/all_gather_rms_norm/zone_timeline.py <dir>/.logs/profile_log_device.csv
"""

import ttnn

if hasattr(ttnn, "Topology"):  # ttnn.operations auto-imports this file during ttnn's own init
    import importlib
    import os

    import pytest

    _op_mod = importlib.import_module("ttnn.operations.all_gather_rms_norm.all_gather_rms_norm")
    from ttnn.operations.all_gather_rms_norm.perf_experiments.parallel_packet_builders import op_copy_descriptor as _c

    _op_mod.create_mesh_program_descriptor = _c.create_mesh_program_descriptor
    from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
    from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases
    from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

    CASES = [_loose_cases()[int(i)] for i in os.environ.get("PPB_CASES", "0").split(",")]

    @pytest.mark.parametrize("inputs,axes,extras", CASES)
    def test_op_zones(inputs, axes, extras, _fabric_mesh_cache):
        mesh = _fabric_mesh_cache.get((1, 4))
        call, check = prepare_all_gather_rms_norm(
            inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes
        )
        check(call())
        for _ in range(5):
            call()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
