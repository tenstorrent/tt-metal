# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The op's two program builders behind one signature, so a descriptor test can pin both.

`ttnn.bringup.rms_norm` builds its program in C++ (device/rms_norm_ttnn_program_factory.cpp), ported
from the Python builder (rms_norm_ttnn_program_descriptor.py), which stays as the reference.  A test
that inspects the SHIPPED program runs on both; a test that flips a Python module knob can only reach
the Python one (the C++ knobs are compile-time constants at their shipped values).
"""

from __future__ import annotations

import ttnn

from ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import create_program_descriptor


def cpp_create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    weight=None,
    bias=None,
    residual=None,
    epsilon=1e-12,
    compute_kernel_config=None,
    program_config=None,
):
    """create_program_descriptor()'s signature, built by the C++ host side."""
    return ttnn._ttnn.operations.bringup._rms_norm_ttnn_program_descriptor(
        input_tensor,
        output_tensor,
        weight=weight,
        bias=bias,
        residual=residual,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        subblock_w=(program_config.subblock_w if program_config is not None else 0),
    )


BUILDERS = {"python": create_program_descriptor, "cpp": cpp_create_program_descriptor}
BUILDER_IDS = list(BUILDERS)
