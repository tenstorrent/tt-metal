# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bake-off (perf idea E2): fuse groupnorm_sc_N_1_HW_C's pass-1 chunk work
(square + REDUCE_COL(x) + REDUCE_COL(x^2)) into one chain + ONE REDUCE_COL over [x | x^2]."""
