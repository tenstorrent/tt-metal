# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p gw_grad_plugin, PYTHONPATH=<this dir>): the REAL op's entry point runs the graduated program
descriptor (grad/, = graduation.patch applied; real kernels), so the unmodified golden suite gates the new rule."""
import ttnn.operations.mhc_pre.mhc_pre as real_op
from grad_loader import grad_pd

real_op.create_program_descriptor = grad_pd.create_program_descriptor


def pytest_report_header(config):
    return f"gw_grad_plugin: mhc_pre descriptor -> {grad_pd.__file__} (kernels {grad_pd.KERNEL_DIR})"
