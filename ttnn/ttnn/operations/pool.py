# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

# TODO(nuked-op pool): pool ops (max_pool2d, avg_pool2d, global_avg_pool2d, upsample,
# grid_sample, rotate) and experimental adaptive_*_pool2d were removed for eval. This
# module previously defined their golden functions and attached them via
# ttnn.attach_golden_function(...). Those attaches referenced now-removed ttnn ops and
# crashed at import, so they have been stripped. Restore the golden functions and the
# attach calls once the pool ops are recreated.


# TODO(nuked-op pool): grid_sample binding removed for eval. Kept so `import ttnn`
# (which imports this name in ttnn/__init__.py) still works; calling it at runtime
# raises because the C++ binding no longer exists. Restore on recreate.
def prepare_grid_sample_grid(*args, **kwargs):
    """
    Precomputes grid sample data for optimized kernel execution.

    NOTE: stubbed for eval — the underlying ttnn._ttnn.operations.pool.prepare_grid_sample_grid
    binding was removed along with the grid_sample op.
    """
    return ttnn._ttnn.operations.pool.prepare_grid_sample_grid(*args, **kwargs)
