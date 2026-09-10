# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from ttnn._ttnn.layer_completion import (
    LayerCompletionQueue,
    LayerCompletionRouter,
)

__all__ = ["LayerCompletionQueue", "LayerCompletionRouter"]

# Test-only scheduler stand-in: registered only in TTNN_BUILD_TESTS builds, so it is absent from a
# shipped wheel. Importing it must not take the production ring/router down with it.
try:
    from ttnn._ttnn.layer_completion import LayerCompletionConsumer
except ImportError:
    pass
else:
    __all__.append("LayerCompletionConsumer")
