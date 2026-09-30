# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p skb_op_plugin, PYTHONPATH=<this dir>): the REAL op's entry point with the program descriptor +
kernels of ../op/ (the graduated fast Sinkhorn = graduation.patch), so the unmodified golden / unit suites gate it."""
import importlib as _il

real_op = _il.import_module("ttnn.operations.mhc_pre.mhc_pre")
import ttnn.operations.mhc_pre.perf_experiments.sinkhorn_sfpu_fast.op.mhc_pre_program_descriptor as grad_pd

real_op.create_program_descriptor = grad_pd.create_program_descriptor


def pytest_report_header(config):
    return f"skb_op_plugin: mhc_pre descriptor -> {grad_pd.__file__} (kernels {grad_pd.KERNEL_DIR})"
