# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln_lanczos — multivariate log-gamma at p=4 via Lanczos.

Computes ``torch.special.multigammaln(x, p=4)`` as a single fused TTNN kernel.
"""

from .multigammaln_lanczos import multigammaln_lanczos

__all__ = ["multigammaln_lanczos"]
