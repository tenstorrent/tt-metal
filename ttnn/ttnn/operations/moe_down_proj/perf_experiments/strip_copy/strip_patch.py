# strip_copy pytest plugin (-p strip_patch): re-routes the L layout x strip copy for the perf bench.
#   MOE_STRIP=base  op's layout (rotating sender sends xa + xb)
#   MOE_STRIP=fwd   forwarder layout: X_MAIN senders = rect row minus core (g, g); X_STRIP fixed sender (g, g)
# Use with MOE_KERNEL_DIR=<kernels_base | kernels_fwd> (the fwd layout needs kernels_fwd).
import os

import ttnn
from ttnn.operations.moe_down_proj import moe_down_proj_program_descriptor as pd


# MOE_FWD_COL=diag (default): forwarder (g, g), the rect core straight above the strip core; =last: (gx - 1, g).
def _forwarder_x(grid, g):
    return grid.x - 1 if os.environ.get("MOE_FWD_COL", "diag") == "last" else g


def _l_layout_fwd(grid):
    layout = _orig_l_layout(grid)
    if layout is None:
        return None
    h = layout.num_m_groups
    rect = pd._crs(0, 0, grid.x - 1, h - 1)
    strip = pd._crs(0, grid.y - 1, h - 1, grid.y - 1)
    others = [
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
        for y in range(h)
        for x in range(grid.x)
        if x != _forwarder_x(grid, y)
    ]
    senders = ttnn.CoreRangeSet(others)
    if grid.x - 1 > 1:
        x_sender = ttnn.McastRotatingSenderConfig(senders, sender_order=ttnn.McastCoreOrder.RowMajor)
    else:
        x_sender = ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(1 - _forwarder_x(grid, g), g) for g in range(h)])
    layout.channels[pd.X_MAIN] = (rect, grid.x, x_sender, ttnn.McastCoreOrder.RowMajor)
    layout.channels[pd.X_STRIP] = (
        strip,
        1,
        ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(_forwarder_x(grid, g), g) for g in range(h)]),
        ttnn.McastCoreOrder.RowMajor,
    )
    return layout


_orig_l_layout = pd._l_layout
if os.environ.get("MOE_STRIP", "base") == "fwd":
    pd._l_layout = _l_layout_fwd
    pd.LAYOUTS["L"] = _l_layout_fwd

if os.environ.get("MOE_KERNEL_DIR"):
    from pathlib import Path

    pd.KERNEL_DIR = Path(os.environ["MOE_KERNEL_DIR"])
