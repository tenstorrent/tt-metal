# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""writer_dual_risc — isolated bake-off harness.

Correctness (bit identity — tilize does no arithmetic) needs a PLAIN run:
    scripts/run_safe_pytest.sh tests_or_this_file...
Actually this file lives under perf_experiments/, so run it directly:
    scripts/run_safe_pytest.sh \
        ttnn/ttnn/operations/tilize/perf_experiments/writer_dual_risc/test_writer_dual_risc.py

Perf numbers (device kernel ns per row, in execution order) need `--profile`:
    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/tilize/perf_experiments/writer_dual_risc/test_writer_dual_risc.py
then read generated/profiler/reports/*/ops_perf_results*.csv, column
`DEVICE KERNEL DURATION [ns]` — one row per dispatch, in the order the prints
below name them.
"""


import ttnn

from ttnn.operations.tilize.perf_experiments.writer_dual_risc import bench

FOCUS_SHAPE = (1, 1, 32, 16384)
DOMAIN_SHAPES = [
    ((1, 1, 32, 32768), "wide_short"),
    ((1, 1, 1024, 1024), "square"),
    ((1, 1, 16384, 32), "tall_narrow"),
    ((1, 1, 32, 2048), "small"),
]


def _make_tensors(device, shape):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input, tt_output


def _run_variant(device, shape, variant, label):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    torch_input, tt_input, tt_output = _make_tensors(device, shape)
    plan = bench.derive_geometry(tt_input, tt_output, device)
    if variant == "col_split" and not bench.col_split_applicable(plan):
        print(f"\n[{label} {variant}] {shape}: col_split INEXPRESSIBLE at block_width_tiles={plan.block_width_tiles}")
        return None
    descriptor = bench.build_descriptor(tt_input, tt_output, plan, variant)
    ttnn.generic_op([tt_input, tt_output], descriptor)
    got = ttnn.to_torch(tt_output)
    ok = torch.equal(got, torch_input)
    print(
        f"\n[{label} {variant}] {shape}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"R={plan.tensor_row_blocks} cores={len(plan.assignment)} bit_identical={ok}"
    )
    assert ok, f"{label}/{variant}: tilize is not bit-identical"
    return plan


def test_correctness_focus_all_variants(device):
    for variant in bench.VARIANTS:
        _run_variant(device, FOCUS_SHAPE, variant, "focus")


def test_correctness_domain_all_variants(device):
    for shape, label in DOMAIN_SHAPES:
        for variant in bench.VARIANTS:
            _run_variant(device, shape, variant, label)


def test_perf_focus(device):
    """Profile target: run baseline / col_split / role_swap on the focus shape,
    in this fixed order, for a `--profile` capture."""
    for variant in bench.VARIANTS:
        _run_variant(device, FOCUS_SHAPE, variant, "PERF focus")


def test_perf_domain(device):
    """Profile target: the domain sweep, same fixed order per shape."""
    for shape, label in DOMAIN_SHAPES:
        for variant in bench.VARIANTS:
            _run_variant(device, shape, variant, f"PERF {label}")


def test_perf_decisive(device):
    """THE DECISIVE CHECK. Write-only (no reader, no compute), fixed 8-tile
    per-core payload, active core count swept 4 -> 64. Per-core write time
    FLAT across this sweep => issue-bound (the dual-RISC split idea has real
    room). Per-core write time falling sharply at 4 cores relative to 64
    => the store was DRAM-bandwidth-bound at 64 cores and splitting the issue
    across a second RISC-V cannot help there."""
    for k in (4, 8, 16, 32, 64):
        shape = (32, k * bench.BLOCK_WIDTH_TILES_DECISIVE * 32)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        descriptor = bench.build_decisive_descriptor(out, k)
        ttnn.generic_op([out], descriptor)
        ttnn.synchronize_device(device)
        print(f"\n[PERF decisive] num_active_cores={k} shape={shape}")
