# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin of the issue_path experiment: runs any g6_clone test suite with a variant's kernels.

ISSUE_VARIANT=<name> points the op's program descriptor at perf_experiments/issue_path/kernels_<name>/ (same
CT/RT layout as the op's kernels, host unchanged) for the whole session, without touching the op's files, and
checks that every descriptor built in the session really uses that folder. Load it with
    PYTHONPATH=<folder holding a copy of this file>:$PYTHONPATH scripts/run_safe_pytest.sh <tests> -p issue_path_plugin
(run.sh's copy-to-generated/ rule applies: a module under ttnn/ imported by pytest registers ttnn twice).
"""

from __future__ import annotations

import os
from pathlib import Path

_STATE = {"variant": "", "dir": None, "built": 0, "foreign": []}


def pytest_configure(config):
    variant = os.environ.get("ISSUE_VARIANT", "")
    if not variant:
        return
    import ttnn.operations.g6_clone.g6_clone_program_descriptor as pd

    kernel_dir = Path(os.environ["ISSUE_KERNEL_ROOT"]) / f"kernels_{variant}"
    if not (kernel_dir / "g6_clone_reader.cpp").is_file() or not (kernel_dir / "g6_clone_writer.cpp").is_file():
        raise RuntimeError(f"issue_path_plugin: no variant kernels in {kernel_dir}")
    pd.KERNEL_DIR = kernel_dir
    real_create = pd.create_program_descriptor

    def create_program_descriptor(*args, **kwargs):
        descriptor = real_create(*args, **kwargs)
        _STATE["built"] += 1
        for kernel in descriptor.kernels:
            if not str(kernel.kernel_source).startswith(str(kernel_dir)):
                _STATE["foreign"].append(str(kernel.kernel_source))
        return descriptor

    pd.create_program_descriptor = create_program_descriptor
    # g6_clone.py imported the name at its own import: patch that binding too (through sys.modules: the
    # package attribute g6_clone is the entry function, which shadows the submodule of the same name).
    import importlib
    import sys

    importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    sys.modules["ttnn.operations.g6_clone.g6_clone"].create_program_descriptor = create_program_descriptor
    _STATE.update(variant=variant, dir=kernel_dir)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _STATE["variant"]:
        return
    terminalreporter.write_line(
        f"ISSUE_PLUGIN variant={_STATE['variant']} dir={_STATE['dir']} descriptors_built={_STATE['built']} "
        f"foreign_kernel_sources={len(_STATE['foreign'])}"
    )
    for source in sorted(set(_STATE["foreign"]))[:10]:
        terminalreporter.write_line(f"ISSUE_PLUGIN foreign source: {source}")
