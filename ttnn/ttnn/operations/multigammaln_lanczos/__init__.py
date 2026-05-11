# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln_lanczos — multivariate log-gamma at order p = 4, implemented as a
faithful translation of the Lanczos 6-term polynomial recipe into a single fused
TTNN kernel (no SFPU lgamma helpers).
"""

from .multigammaln_lanczos import multigammaln_lanczos

__all__ = ["multigammaln_lanczos"]
