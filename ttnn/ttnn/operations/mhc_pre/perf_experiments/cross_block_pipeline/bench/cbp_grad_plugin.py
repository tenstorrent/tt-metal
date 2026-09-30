# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p cbp_grad_plugin, PYTHONPATH=<this dir>): runs the REAL op's entry point with the graduated
program descriptor + kernels of grad/ (the content of graduation.patch), so the unmodified golden suite gates it."""
import ttnn.operations.mhc_pre.mhc_pre as real_op
import ttnn.operations.mhc_pre.perf_experiments.cross_block_pipeline.grad.mhc_pre_program_descriptor as grad_pd

real_op.create_program_descriptor = grad_pd.create_program_descriptor


def pytest_report_header(config):
    return f"cbp_grad_plugin: mhc_pre descriptor -> {grad_pd.__file__} (kernels {grad_pd.KERNEL_DIR})"
