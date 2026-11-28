# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: run any g6_clone suite with the fused one-kernel descriptor installed.

    FUSED_VARIANT=ncrisc_dualw PYTHONPATH=<dir holding this file> \
        scripts/run_safe_pytest.sh <tests> -p fused_plugin

The op's files are untouched; the swap lives only in this pytest process. The terminal summary
reports how many fused descriptors were built, so a run that never reached the swap shows 0."""

import importlib
import os

_BUILT = {"count": 0, "kernels": set()}


def pytest_sessionstart(session):
    variant = os.environ.get("FUSED_VARIANT", "")
    if not variant:
        raise RuntimeError("fused_plugin: set FUSED_VARIANT")
    fused = importlib.import_module("ttnn.operations.g6_clone.perf_experiments.fused_single_risc.fused_descriptor")
    fused.install(variant)
    dm = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")
    op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    create = dm.create_program_descriptor

    def counted(input_tensor, output_tensor, plan):
        descriptor = create(input_tensor, output_tensor, plan)
        _BUILT["count"] += 1
        for kernel in descriptor.kernels:
            _BUILT["kernels"].add(os.path.basename(str(kernel.kernel_source)))
        return descriptor

    dm.create_program_descriptor = counted
    op.create_program_descriptor = counted


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"FUSED_PLUGIN variant={os.environ.get('FUSED_VARIANT', '')} descriptors_built={_BUILT['count']} "
        f"kernels={sorted(_BUILT['kernels'])}"
    )
