# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from . import expression
from ttnn._ttnn.operations.experimental.materialize.prim import (
    materialize,
)
import sys

# merge exports with experimental_loader
if "ttnn.experimental" in sys.modules:
    experimental = sys.modules["ttnn.experimental"]
    setattr(experimental, "materialize", materialize)
    setattr(experimental, "expression", expression)
    del experimental

__all__ = ["expression", "materialize"]
del sys
