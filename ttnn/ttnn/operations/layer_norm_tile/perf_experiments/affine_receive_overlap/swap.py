# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin (-p swap, PYTHONPATH=<this dir>): LN_VARIANT=<v> points the op's KERNEL_DIR at kernels/<v>.

Patches the descriptor module the public entry actually resolves (ttnn may hold a second copy).
"""

import os
from pathlib import Path

from ttnn.operations.layer_norm_tile import layer_norm as _layer_norm

VARIANT = os.environ.get("LN_VARIANT", "base")
_pd_globals = _layer_norm.__globals__["create_program_descriptor"].__globals__
_pd_globals["KERNEL_DIR"] = Path(__file__).parent / "kernels" / VARIANT
print(f"\nARO_SWAP variant={VARIANT} KERNEL_DIR={_pd_globals['KERNEL_DIR']}")
