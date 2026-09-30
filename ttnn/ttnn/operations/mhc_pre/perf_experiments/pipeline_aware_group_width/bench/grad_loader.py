# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Loads grad/mhc_pre_program_descriptor.py (= the real descriptor + graduation.patch) as a module whose kernels
are the REAL op's (the patch is host-only)."""
import importlib.util
import os

import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as real_pd

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grad", "mhc_pre_program_descriptor.py")
_spec = importlib.util.spec_from_file_location("mhc_pre_gw_grad_pd", _P)
grad_pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grad_pd)
grad_pd.KERNEL_DIR = real_pd.KERNEL_DIR
