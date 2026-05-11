# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln — multivariate log-gamma at order p = 4.

    multigammaln(a) = lgamma(a) + lgamma(a - 0.5) + lgamma(a - 1.0) + lgamma(a - 1.5)
                    + 3 * log(pi)

Matches torch.special.multigammaln(x, p=4).
"""

from .multigammaln import multigammaln

__all__ = ["multigammaln"]
