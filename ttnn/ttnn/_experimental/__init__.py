# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Internal experimental modules to be dynamically added to ttnn.experimental."""

from . import moe_compute_utils
from . import disaggregation

__all__ = [
    "moe_compute_utils",
    "disaggregation",
]
