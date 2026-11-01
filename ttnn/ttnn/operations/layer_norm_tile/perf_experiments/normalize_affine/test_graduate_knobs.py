# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Graduated affine_block kernel (graduate/) on the op's knob-test shapes (multi-block, ragged, h_non_aligned) at
1, 3 and 4 DEST lanes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/localdev/kgrujcic/tt-metal-codegen/tests/ttnn/unit_tests/operations/layer_norm_tile")
from test_layer_norm_tile_stream_knobs import CASES, descriptor, run_and_check  # noqa: E402

GRADUATE = Path(__file__).parent / "graduate"


@pytest.mark.parametrize("lanes", [1, 3, 4])
@pytest.mark.parametrize("shape,buf,with_residual", CASES)
def test_graduate(device, monkeypatch, lanes, shape, buf, with_residual):
    monkeypatch.setattr(descriptor, "KERNEL_DIR", GRADUATE)
    monkeypatch.setattr(descriptor, "COMPUTE_DEST_LANES", lanes)
    run_and_check(device, shape, with_residual=with_residual, input_buf=buf, output_buf=buf)
