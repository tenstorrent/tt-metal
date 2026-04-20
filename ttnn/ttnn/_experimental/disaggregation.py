# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregation operations for ttnn.experimental."""

from ttnn._ttnn.operations.experimental import disaggregation as _disaggregation

# Re-export disaggregation functionality
__all__ = dir(_disaggregation)

# Make all disaggregation functions/classes available at module level
for attr_name in dir(_disaggregation):
    if not attr_name.startswith("_"):
        globals()[attr_name] = getattr(_disaggregation, attr_name)
