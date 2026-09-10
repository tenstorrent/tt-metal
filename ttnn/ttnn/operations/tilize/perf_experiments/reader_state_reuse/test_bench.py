# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_state_reuse isolated bake-off -- correctness + device-ns harness.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/reader_state_reuse/test_bench.py

Correctness (bit identity vs torch, tilize's own contract -- no arithmetic
happens in this stage) for every (variant, shape) cell:

    scripts/run_safe_pytest.sh --run-all .../test_bench.py -k correctness

Device ns (one dispatch per case, execution order = parametrize order below):

    scripts/run_safe_pytest.sh --profile .../test_bench.py -k device_ns

then read `generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv`, column
`DEVICE KERNEL DURATION [ns]`, and the zone breakdown via
`python3 ttnn/ttnn/operations/tilize/perf_experiments/zone_report.py` on the
same run's `profile_log_device.csv` for the reserve/issue/barrier split.
"""

import pytest

from ttnn.operations.tilize.perf_experiments.reader_state_reuse import bench

FOCUS = "focus_1x1x32x16384"
SHAPES = list(bench.DERIVED_BLOCKS)
CASES = [(variant, shape) for shape in SHAPES for variant in bench.VARIANTS]


@pytest.mark.parametrize("variant,shape", CASES, ids=[f"{v}-{s}" for v, s in CASES])
def test_correctness(device, variant, shape):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    rows, row_bytes, nrg, nwc = bench.DERIVED_BLOCKS[shape]
    got, expected = bench.run_variant(
        device, variant, rows_per_block=rows, row_bytes=row_bytes, row_groups_used=nrg, w_chunks_used=nwc
    )
    assert torch.equal(got, expected), f"{variant}/{shape}: reader_state_reuse bench is not bit-identical"


# Same cases, no assert -- this is the ordered sequence a `--profile` run reads
# off the CSV. Kept separate from test_correctness so a `--profile` run's CSV
# rows are exactly this list, in this order, with no interleaved noise.
@pytest.mark.parametrize("variant,shape", CASES, ids=[f"{v}-{s}" for v, s in CASES])
def test_device_ns(device, variant, shape):
    rows, row_bytes, nrg, nwc = bench.DERIVED_BLOCKS[shape]
    bench.run_variant(device, variant, rows_per_block=rows, row_bytes=row_bytes, row_groups_used=nrg, w_chunks_used=nwc)
