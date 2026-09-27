# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `program_config` variants this op accepts (D-06).

Two alternatives, told apart by which fields they carry:
`compute_with_storage_grid_size` and the blocking beside it mark the sharded
multi-core variant; the three bare algorithm flags mark the default
(interleaved) one.  Field names, types, defaults and keyword-only construction
match the specification target's, so a caller's existing construction is the
same call against either op.

These live on the OP's public surface rather than in a test helper because a
caller reaches them: the argument is useless without a type to build.  They
were a fixture-local stand-in until the reference suite showed no real caller
could construct one.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn


@dataclass(kw_only=True)
class RMSNormDefaultProgramConfig:
    """The interleaved-input variant: three algorithm flags, nothing else."""

    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False


@dataclass(kw_only=True)
class RMSNormShardedMultiCoreProgramConfig:
    """The sharded variant: the caller's grid and blocking, plus the same flags."""

    compute_with_storage_grid_size: ttnn.CoreCoord
    subblock_w: int
    block_h: int
    block_w: int
    inplace: bool
    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False

    def __post_init__(self):
        # The target's constructor is bound with a CoreCoord parameter, and its
        # binding layer converts a 2-sequence for free -- so callers pass
        # `(x, y)` and `[x, y]` as readily as a CoreCoord, and a great many do.
        # A Python class gets no such conversion, so do it here or every such
        # call dies on `.x` deep inside the descriptor.
        grid = self.compute_with_storage_grid_size
        if not hasattr(grid, "x"):
            x, y = grid
            object.__setattr__(self, "compute_with_storage_grid_size", ttnn.CoreCoord(x, y))
