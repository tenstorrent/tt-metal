# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Point the real layer_norm_tile op at this experiment's kernels and pick the normalize/affine variant.

LN_NA=<name> selects the variant (one per pytest process: generic_op caches programs). The variant id rides
on the compute compile-time arg the op already carries for FUSE_NORMALIZE_AFFINE (index 16).
"""

import os
from pathlib import Path

from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd

VARIANTS = {"baseline": 0, "parked": 1, "seed": 2, "parked_lean": 3, "seed_raw": 4, "seed_il": 5, "seed_pipe": 6}
DIAG = {"d_mul_col": 10, "d_mul_row": 11, "d_add_row": 12, "d_copy": 13, "d_bcast_row": 14, "d_seed_2xinit": 15}


def install():
    name = os.environ.get("LN_NA", "baseline")
    if name == "graduate":
        # The proposed real-op compute kernel (graduate/), driven by the unchanged host.
        _pd.KERNEL_DIR = Path(__file__).parent / "graduate"
        return name
    _pd.KERNEL_DIR = Path(__file__).parent / "kernels"
    _pd.FUSE_NORMALIZE_AFFINE = VARIANTS[name]
    return name
