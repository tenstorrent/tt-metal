# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_x_first bake-off driver.

One test per regime; inside, every reader ordering runs once on the op's exact geometry and is
gated byte-for-byte (x and gamma vs torch, scaler vs the baseline ordering's scaler). Perf is read
from the profiler CSV of a `scripts/run_safe_pytest.sh --profile ... -k <regime>` run: the rows are in
execution order = ORDERS order below.

    scripts/run_safe_pytest.sh --profile ttnn/ttnn/operations/rms_norm/perf_experiments/reader_x_first/test_reader_x_first.py --import-mode=prepend -k tile7168
"""

from __future__ import annotations

import pytest
import ttnn

# Run with `--import-mode=prepend` (this dir is then on sys.path): the repo's default importlib mode
# would name this module ttnn.ttnn.…, importing the ttnn package twice and double-registering ops.
from bench import ORDER_NAMES, TILE, make_inputs, num_active_cores, run_variant

TL, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

# regime -> (shape, x layout, gamma layout (None = no gamma), orders, expected active cores)
ALL_TILE = (0, 1, 2, 3, 4, 5, 6, 7, 8)
RM_ORDERS = (0, 1, 4, 7, 8)  # on the stick-helper path 2 == 1 and 5 == 4; 3 / 6 are not expressible
REGIMES = {
    "tile7168_focus": ((1, 1, 32, 7168), TL, TL, ALL_TILE, 26),
    "tile5120_decode": ((1, 1, 32, 5120), TL, TL, ALL_TILE, 10),
    "tile1024_prefill": ((1, 1, 8320, 1024), TL, TL, ALL_TILE, 130),
    "rm7168_rmgamma": ((1, 1, 32, 7168), RM, RM, RM_ORDERS, 26),
    "rm7168_tilegamma": ((1, 1, 32, 7168), RM, TL, RM_ORDERS, 26),
    "tile7168_nogamma": ((1, 1, 32, 7168), TL, None, (0, 2, 4, 7, 8), 26),
}


def _bits(t: torch.Tensor) -> torch.Tensor:
    import torch

    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("regime", list(REGIMES.keys()))
def test_reader_orderings(device, regime):
    import torch

    shape, x_layout, g_layout, orders, expect_cores = REGIMES[regime]
    x, x_t, gamma, g_t = make_inputs(device, shape, x_layout, g_layout)
    n_active, blocking = num_active_cores(device, x, gamma)
    wcs = sorted({r.core_w_tiles for r in blocking.all_roles if r.is_active}, reverse=True)
    print(
        f"\n[{regime}] regime={blocking.regime} active_cores={n_active} rect={blocking.rect_a}x{blocking.rect_b} "
        f"row_groups={blocking.num_row_groups} block_rows={blocking.block_rows} core_w_tiles={wcs} Wt={blocking.tensor_w_tiles} Rt={blocking.tensor_row_tiles}"
    )
    assert n_active == expect_cores, f"geometry drifted: {n_active} active cores, expected {expect_cores}"

    baseline_scaler = None
    for order in orders:
        ox, og, osc = run_variant(device, x, gamma, order, n_active)
        name = ORDER_NAMES[order]
        # x: byte-identical to the input (every tile / stick landed, at the right id)
        assert torch.equal(_bits(ox), _bits(x_t)), f"{regime}/{name}: x CB contents differ from input"
        # gamma
        if gamma is not None:
            if g_layout == RM:
                assert torch.equal(_bits(og[0, 0, 0]), _bits(g_t[0, 0, 0])), f"{regime}/{name}: RM gamma row 0 differs"
                assert torch.all(og[0, 0, 1:TILE] == 0), f"{regime}/{name}: RM gamma rows 1..31 not zero-filled"
            else:
                assert torch.equal(_bits(og), _bits(g_t)), f"{regime}/{name}: TILE gamma CB contents differ"
        # scaler: one bf16 tile per core, only {0, 1.0}, and identical to the baseline ordering's tile
        vals = osc.float()
        assert torch.all((vals == 0) | (vals == 1.0)), f"{regime}/{name}: scaler tile has values outside {{0,1}}"
        assert torch.all(vals.sum(dim=(-1, -2)) > 0), f"{regime}/{name}: an empty scaler tile"
        if baseline_scaler is None:
            baseline_scaler = osc
        else:
            assert torch.equal(_bits(osc), _bits(baseline_scaler)), f"{regime}/{name}: scaler differs from baseline"
        print(f"[{regime}] order {order} ({name}): CB contents byte-identical -> PASS")
